
import random

import numpy as np


ARCHITECTURE_VERSION = "multiscale-rescnn-bilstm-attention-v2-keras216"
_KERAS_SESSION = None


def require_legacy_stack():

    try:
        import tensorflow as tf
        import keras
        from keras import backend as K
    except ImportError as exc:
        raise RuntimeError(
            "This operation requires tensorflow-gpu==1.13.1 and keras==2.1.6. "
            "Run it in the specified legacy Conda environment."
        ) from exc

    if K.backend() != "tensorflow":
        raise RuntimeError("The Keras backend must be tensorflow; check ~/.keras/keras.json.")
    return tf, keras, K


def configure_legacy_session(seed=42, allow_growth=True):

    global _KERAS_SESSION
    tf, keras, K = require_legacy_stack()
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)



    try:
        K.clear_session()
    except Exception:
        pass
    if _KERAS_SESSION is not None:
        try:
            _KERAS_SESSION.close()
        except Exception:
            pass
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = bool(allow_growth)
    _KERAS_SESSION = tf.Session(config=config)
    try:
        from keras.backend.tensorflow_backend import set_session
        set_session(_KERAS_SESSION)
    except ImportError:

        K.set_session(_KERAS_SESSION)
    return tf, keras, K


def compile_network(model, learning_rate=1e-3):

    _, keras, _ = require_legacy_stack()
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _conv_norm_activation(keras, inputs, filters, kernel_size, name, dilation=1):

    layers = keras.layers
    regularizer = keras.regularizers.l2(1e-4)
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        use_bias=False,
        name=name + "_conv",
    )(inputs)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.Activation("relu", name=name + "_relu")(x)


def _residual_block(keras, inputs, filters, kernel_size, dilation, name):

    layers = keras.layers
    shortcut = inputs
    if int(inputs.shape[-1]) != filters:
        shortcut = layers.Conv1D(
            filters, 1, padding="same", use_bias=False, name=name + "_projection"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=name + "_projection_bn")(shortcut)

    x = _conv_norm_activation(keras, inputs, filters, kernel_size, name + "_first", dilation)
    x = layers.SpatialDropout1D(0.15, name=name + "_dropout")(x)
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        use_bias=False,
        name=name + "_second_conv",
    )(x)
    x = layers.BatchNormalization(name=name + "_second_bn")(x)
    x = layers.Add(name=name + "_add")([shortcut, x])
    return layers.Activation("relu", name=name + "_output")(x)


def _attention_softmax(logits):

    from keras import backend as K
    scores = K.squeeze(logits, axis=-1)
    weights = K.softmax(scores)
    return K.expand_dims(weights, axis=-1)


def _attention_shape(input_shape):
    return input_shape


def _sum_over_time(weighted_sequence):
    from keras import backend as K
    return K.sum(weighted_sequence, axis=1)


def _sum_over_time_shape(input_shape):
    return (input_shape[0], input_shape[2])


def build_network(sequence_length, channel_count, learning_rate=1e-3, seed=42):

    if sequence_length < 2 or channel_count < 1:
        raise ValueError("sequence_length must be at least 2 and channel_count must be at least 1.")

    _, keras, _ = configure_legacy_session(seed)
    layers = keras.layers

    inputs = layers.Input(
        shape=(sequence_length, channel_count),
        name="encoded_site_window",
    )


    scale_3 = _conv_norm_activation(keras, inputs, 64, 3, "scale3")
    scale_5 = _conv_norm_activation(keras, inputs, 64, 5, "scale5")
    scale_9 = _conv_norm_activation(keras, inputs, 64, 9, "scale9")
    x = layers.Concatenate(name="multiscale_concat")([scale_3, scale_5, scale_9])
    x = _conv_norm_activation(keras, x, 128, 1, "multiscale_fusion")

    x = _residual_block(keras, x, 128, 3, 1, "residual_local_1")
    x = _residual_block(keras, x, 128, 5, 1, "residual_local_2")
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="sequence_downsample")(x)
    x = _residual_block(keras, x, 192, 3, 2, "residual_dilated_1")
    x = _residual_block(keras, x, 192, 5, 2, "residual_dilated_2")



    if hasattr(layers, "CuDNNLSTM"):
        recurrent = layers.CuDNNLSTM(64, return_sequences=True, name="context_lstm")
    else:
        recurrent = layers.LSTM(64, return_sequences=True, name="context_lstm")
    x = layers.Bidirectional(recurrent, name="bidirectional_context")(x)
    x = layers.SpatialDropout1D(0.20, name="context_dropout")(x)



    attention_logits = layers.Dense(1, activation="tanh", name="attention_score")(x)
    attention_weights = layers.Lambda(
        _attention_softmax,
        output_shape=_attention_shape,
        name="attention_weights",
    )(attention_logits)
    weighted_sequence = layers.Multiply(name="attention_multiply")([x, attention_weights])
    attention_pool = layers.Lambda(
        _sum_over_time,
        output_shape=_sum_over_time_shape,
        name="attention_pool",
    )(weighted_sequence)

    average_pool = layers.GlobalAveragePooling1D(name="average_pool")(x)
    maximum_pool = layers.GlobalMaxPooling1D(name="maximum_pool")(x)
    x = layers.Concatenate(name="global_context")([attention_pool, average_pool, maximum_pool])

    x = layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = layers.Dropout(0.35, name="head_dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="head_dense_2")(x)
    x = layers.Dropout(0.20, name="head_dropout_2")(x)
    probability = layers.Dense(1, activation="sigmoid", name="cleavage_probability")(x)

    model = keras.models.Model(
        inputs=inputs,
        outputs=probability,
        name="CleavageSiteMultiScaleNetKeras216",
    )
    return compile_network(model, learning_rate=learning_rate)


def initialize_transfer_network(
    sequence_length,
    channel_count,
    source_weights,
    unfreeze_last=4,
    learning_rate=2e-4,
    seed=42,
):

    if unfreeze_last < 1:
        raise ValueError("unfreeze_last must be at least 1.")

    model = build_network(sequence_length, channel_count, learning_rate, seed)
    model.load_weights(source_weights)
    for layer in model.layers:
        layer.trainable = False

    weighted_layers = [layer for layer in model.layers if layer.weights]
    for layer in weighted_layers[-unfreeze_last:]:
        layer.trainable = True


    return compile_network(model, learning_rate=learning_rate)
