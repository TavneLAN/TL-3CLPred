
import csv
from datetime import datetime
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np

from . import __version__
from .features import (
    FEATURE_SCHEMA_VERSION,
    channel_names,
    encode_sequences,
    normalize_encoding_name,
)
from .network import ARCHITECTURE_VERSION, build_network, initialize_transfer_network, require_legacy_stack
from .reporting import (
    binary_metrics,
    make_training_monitor,
    plot_attention,
    plot_data_overview,
    plot_ensemble_history,
    plot_feature_heatmap,
    plot_training_history,
    save_evaluation_bundle,
    save_model_description,
    summarize_history,
    write_history,
)
from .sequence_data import SiteWindow, build_site_windows, parse_focus_residues, read_marked_fasta


SCHEMA_VERSION = 2


def _prefix_path(prefix):
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _config_path(prefix):
    return Path(f"{prefix}.json")


def _weight_path(prefix, member_index):
    return Path(f"{prefix}.member_{member_index:03d}.weights.h5")


def _history_path(prefix, member_index):
    return Path(f"{prefix}.member_{member_index:03d}.history.csv")


def _member_prefix(prefix, member_index):
    return Path(f"{prefix}.member_{member_index:03d}")


def _artifact_files(prefix):

    prefix = Path(prefix)
    start = prefix.name + "."
    names = [
        path.name for path in prefix.parent.iterdir()
        if path.name.startswith(start)
    ]
    config_name = _config_path(prefix).name
    if config_name not in names:
        names.append(config_name)
    return sorted(names)


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _runtime_manifest():

    tf, keras, K = require_legacy_stack()
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except Exception:
        matplotlib_version = "unavailable"
    try:
        import pandas
        pandas_version = pandas.__version__
    except Exception:
        pandas_version = "unavailable"
    try:
        gpu_available = bool(tf.test.is_gpu_available(cuda_only=True))
        gpu_device = tf.test.gpu_device_name() or None
    except Exception:
        gpu_available = None
        gpu_device = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas_version,
        "matplotlib": matplotlib_version,
        "keras": getattr(keras, "__version__", "unknown"),
        "keras_backend": K.backend(),
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "gpu_available": gpu_available,
        "gpu_device": gpu_device,
    }


def load_config(prefix):
    path = _config_path(prefix)
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported configuration schema_version: {config.get('schema_version')}")
    if config.get("architecture") != ARCHITECTURE_VERSION:
        raise ValueError(f"Incompatible model architecture: {config.get('architecture')}")
    if config.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Incompatible feature schema: {config.get('feature_schema')}")
    return config


def _resolve_weight_files(prefix, config):
    directory = _config_path(prefix).parent
    paths = [directory / name for name in config.get("weight_files", [])]
    if not paths:
        raise ValueError("The model configuration has no weight_files.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Weight files not found: " + ", ".join(missing))
    return paths


def load_examples(fasta_path, focus_residues, radius):

    records = read_marked_fasta(fasta_path)
    focus = parse_focus_residues(focus_residues)
    examples = build_site_windows(records, focus, radius=radius)
    return examples, [record.identifier for record in records]


def _labels(examples):
    return np.asarray([example.label for example in examples], dtype=np.float32)


def _validate_binary_training_labels(labels):
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            "Training and stratified validation each require at least two positive and two negative samples; "
            f"current positive={positive_count}, negative={negative_count}."
        )


def stratified_split(labels, validation_fraction, seed):

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    _validate_binary_training_labels(labels)

    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for class_value in (0.0, 1.0):
        indices = np.flatnonzero(labels == class_value)
        rng.shuffle(indices)
        validation_count = int(round(len(indices) * validation_fraction))
        validation_count = min(max(1, validation_count), len(indices) - 1)
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def grouped_split(labels, group_ids, validation_fraction, seed):

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    _validate_binary_training_labels(labels)
    groups = np.asarray(group_ids, dtype=object)
    if groups.shape != labels.shape:
        raise ValueError("The number of group_ids must match the number of labels.")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Protein-level splitting requires at least two distinct FASTA records.")
    validation_group_count = int(round(len(unique_groups) * validation_fraction))
    validation_group_count = min(max(1, validation_group_count), len(unique_groups) - 1)
    rng = np.random.default_rng(seed)

    for _ in range(500):
        shuffled_groups = rng.permutation(unique_groups)
        validation_groups = frozenset(shuffled_groups[:validation_group_count].tolist())
        validation_mask = np.fromiter(
            (group in validation_groups for group in groups), dtype=bool, count=len(groups)
        )
        validation_indices = np.flatnonzero(validation_mask)
        train_indices = np.flatnonzero(~validation_mask)
        if len(np.unique(labels[train_indices])) == 2 and len(np.unique(labels[validation_indices])) == 2:
            rng.shuffle(train_indices)
            rng.shuffle(validation_indices)
            return train_indices, validation_indices

    raise ValueError(
        "Unable to create a protein-level train/validation split where both sets contain positive and negative samples; "
        "increase the number of proteins with positive samples or adjust validation-fraction."
    )


def balanced_bootstrap(labels, seed):

    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Balanced bootstrap requires both positive and negative samples.")

    sample_count = min(len(positive), len(negative))
    chosen_positive = rng.choice(positive, size=sample_count, replace=False)
    chosen_negative = rng.choice(negative, size=sample_count, replace=False)
    selected = np.concatenate([chosen_positive, chosen_negative])
    rng.shuffle(selected)
    return selected


def sequential_balanced_blocks(labels, seed, return_dropped=False):

    labels = np.asarray(labels)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Sequential balanced blocks require both positive and negative samples.")

    rng = np.random.default_rng(seed)
    negative = rng.permutation(negative)
    blocks = []
    full_block_count = len(negative) // len(positive)

    if full_block_count == 0:

        positive_block = rng.choice(positive, size=len(negative), replace=False)
        selected = np.concatenate([positive_block, negative])
        rng.shuffle(selected)
        blocks.append(selected)
        dropped_negative = np.asarray([], dtype=np.int64)
    else:
        usable_negative_count = full_block_count * len(positive)
        dropped_negative = negative[usable_negative_count:].astype(np.int64, copy=False)
        for block_index in range(full_block_count):
            start = block_index * len(positive)
            negative_block = negative[start:start + len(positive)]
            selected = np.concatenate([positive.copy(), negative_block])
            rng.shuffle(selected)
            blocks.append(selected)

    if return_dropped:
        return blocks, dropped_negative
    return blocks


def _make_global_best_checkpoint(keras, weight_path, verbose=1):

    class GlobalBestCheckpoint(keras.callbacks.Callback):
        def __init__(self, path, callback_verbose):
            super(GlobalBestCheckpoint, self).__init__()
            self.path = str(path)
            self.verbose = int(callback_verbose)
            self.best = float("inf")
            self.best_epoch = None
            self.best_block = None
            self.current_block = None
            self.save_count = 0

        def set_block(self, block_index):
            self.current_block = int(block_index)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if "val_loss" not in logs:
                raise ValueError("global-best checkpoint could not find val_loss.")
            current = float(logs["val_loss"])
            if not np.isfinite(current):
                return
            if current < self.best:
                previous = self.best
                self.best = current
                self.best_epoch = int(epoch) + 1
                self.best_block = (
                    None if self.current_block is None else self.current_block + 1
                )
                self.save_count += 1
                self.model.save_weights(self.path, overwrite=True)
                if self.verbose:
                    previous_text = (
                        "inf" if not np.isfinite(previous)
                        else "{0:.6f}".format(previous)
                    )
                    print(
                        "\nEpoch {0:05d}: global val_loss improved from {1} "
                        "to {2:.6f}; saving weights to {3}".format(
                            self.best_epoch, previous_text, self.best, self.path
                        )
                    )

        def summary(self):
            return {
                "monitor": "val_loss",
                "mode": "min",
                "scope": "all_sequential_blocks",
                "best_val_loss": (
                    None if not np.isfinite(self.best) else float(self.best)
                ),
                "best_epoch": self.best_epoch,
                "best_block": self.best_block,
                "save_count": int(self.save_count),
                "weight_file": Path(self.path).name,
            }

    return GlobalBestCheckpoint(weight_path, verbose)


def _training_callbacks(
    patience,
    metric_monitor,
    tensorboard_dir,
    global_checkpoint,
):
    _, keras, _ = require_legacy_stack()
    callbacks = [metric_monitor]
    if patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                verbose=1,
            )
        )
    callbacks.extend([
        global_checkpoint,
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        ),
    ])
    return callbacks


def _write_scored_examples(path, examples, indices, scores, threshold=0.5):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "dataset_index", "protein_id", "position_0based", "position_1based",
            "residue", "window_sequence", "label", "probability", "prediction",
        ])
        for dataset_index, score in zip(indices, scores):
            example = examples[int(dataset_index)]
            writer.writerow([
                int(dataset_index), example.protein_id, example.position0, example.position1,
                example.residue, example.sequence, example.label,
                "{0:.8f}".format(float(score)),
                int(float(score) >= threshold),
            ])


def _write_training_blocks(path, examples, blocks):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "block", "dataset_index", "protein_id", "position_0based",
            "position_1based", "residue", "window_sequence", "label",
        ])
        for block_index, indices in enumerate(blocks):
            for dataset_index in indices:
                example = examples[int(dataset_index)]
                writer.writerow([
                    block_index + 1,
                    int(dataset_index),
                    example.protein_id,
                    example.position0,
                    example.position1,
                    example.residue,
                    example.sequence,
                    example.label,
                ])


def _write_dropped_negative_remainder(path, examples, indices):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "dataset_index", "protein_id", "position_0based", "position_1based",
            "residue", "window_sequence", "label", "drop_reason",
        ])
        for dataset_index in indices:
            example = examples[int(dataset_index)]
            writer.writerow([
                int(dataset_index),
                example.protein_id,
                example.position0,
                example.position1,
                example.residue,
                example.sequence,
                example.label,
                "incomplete_negative_remainder",
            ])


def _fit_member(
    model,
    encoded,
    labels,
    training_blocks,
    training_indices,
    x_validation,
    y_validation,
    weight_path,
    member_prefix,
    epochs,
    batch_size,
    patience,
):
    _, keras, _ = require_legacy_stack()
    history_csv_path = Path(str(member_prefix) + ".history.csv")
    history_json_path = Path(str(member_prefix) + ".history.json")
    records = []
    completed_epochs = 0
    elapsed_seconds = 0.0
    global_checkpoint = _make_global_best_checkpoint(
        keras, weight_path, verbose=1
    )




    for block_index, block_indices in enumerate(training_blocks):
        global_checkpoint.set_block(block_index)
        x_block = encoded[block_indices]
        y_block = labels[block_indices]
        positive_count = int(np.sum(y_block == 1))
        negative_count = int(np.sum(y_block == 0))
        print(
            "[sequential block] {0}/{1}, samples={2}, positive={3}, negative={4}".format(
                block_index + 1,
                len(training_blocks),
                len(block_indices),
                positive_count,
                negative_count,
            )
        )
        before_count = len(records)
        metric_monitor = make_training_monitor(
            keras,
            x_block,
            y_block,
            x_validation,
            y_validation,
            batch_size,
            live_csv_path=history_csv_path,
            live_json_path=history_json_path,
            shared_records=records,
            block_index=block_index,
            initial_epoch=completed_epochs,
            elapsed_offset=elapsed_seconds,
        )
        model.fit(
            x_block,
            y_block,
            validation_data=(x_validation, y_validation),
            initial_epoch=completed_epochs,
            epochs=completed_epochs + epochs,
            batch_size=batch_size,
            callbacks=_training_callbacks(
                patience,
                metric_monitor,
                Path(str(member_prefix) + ".tensorboard"),
                global_checkpoint,
            ),
            verbose=2,
            shuffle=True,
        )
        completed_in_block = len(records) - before_count
        completed_epochs += completed_in_block
        if records:
            elapsed_seconds = float(records[-1]["elapsed_seconds"])

    checkpoint_summary = global_checkpoint.summary()
    if checkpoint_summary["best_epoch"] is None or not Path(weight_path).is_file():
        raise RuntimeError("Training finished but no global-best checkpoint was produced.")


    model.load_weights(str(weight_path))
    write_history(history_csv_path, history_json_path, records)
    plot_training_history(
        records,
        Path(str(member_prefix) + ".training_curves.png"),
        "Training history: " + member_prefix.name,
    )


    x_train = encoded[training_indices]
    y_train = labels[training_indices]
    train_scores = model.predict(x_train, batch_size=batch_size, verbose=0).reshape(-1)
    validation_scores = model.predict(
        x_validation, batch_size=batch_size, verbose=0
    ).reshape(-1)
    training_metrics = save_evaluation_bundle(
        Path(str(member_prefix) + ".training"),
        y_train,
        train_scores,
        threshold=0.5,
        title="Best-checkpoint training evaluation: " + member_prefix.name,
    )
    validation_metrics = save_evaluation_bundle(
        Path(str(member_prefix) + ".validation"),
        y_validation,
        validation_scores,
        threshold=0.5,
        title="Best-checkpoint validation evaluation: " + member_prefix.name,
    )


    attention_model = keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("attention_weights").output,
    )
    attention = attention_model.predict(
        x_validation, batch_size=batch_size, verbose=0
    )
    plot_attention(
        attention,
        y_validation,
        Path(str(member_prefix) + ".validation.attention.png"),
    )
    return {
        "history": records,
        "train_scores": train_scores,
        "validation_scores": validation_scores,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "block_count": int(len(training_blocks)),
        "block_sample_counts": [int(len(indices)) for indices in training_blocks],
        "checkpoint_summary": checkpoint_summary,
    }


def _base_config(run_type, focus, radius, encoding, seed, weight_paths):
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "run_type": run_type,
        "architecture": ARCHITECTURE_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "runtime_target": {
            "python": "3.6",
            "keras": "2.1.6",
            "tensorflow_gpu": "1.13.1",
            "numpy": "1.19.2",
            "pandas": "0.25.1",
            "cuda": "10.0.130",
            "cudnn": "7.6.5",
        },
        "focus_residues": list(focus),
        "radius": int(radius),
        "sequence_length": int(2 * radius),
        "encoding": normalize_encoding_name(encoding),
        "channel_count": len(channel_names(encoding)),
        "seed": int(seed),
        "model_count": len(weight_paths),
        "weight_files": [path.name for path in weight_paths],
    }


def train_base(
    input_path,
    output_prefix,
    focus_residues,
    radius=15,
    encoding="physchem-v1",
    model_count=1,
    epochs=100,
    batch_size=256,
    validation_fraction=0.1,
    patience=15,
    learning_rate=1e-3,
    seed=42,
):

    if model_count < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("models, epochs, and batch-size must all be at least 1.")
    run_started_utc = _utc_now()
    run_started_clock = time.time()
    prefix = _prefix_path(output_prefix)
    focus = parse_focus_residues(focus_residues)
    encoding = normalize_encoding_name(encoding)
    examples, protein_ids = load_examples(input_path, focus, radius)
    labels = _labels(examples)
    train_indices, validation_indices = grouped_split(
        labels,
        [example.protein_id for example in examples],
        validation_fraction,
        seed,
    )

    sequences = [example.sequence for example in examples]
    encoded = encode_sequences(sequences, encoding)
    x_validation = encoded[validation_indices]
    y_validation = labels[validation_indices]


    plot_data_overview(
        labels,
        labels[train_indices],
        y_validation,
        Path(str(prefix) + ".data_split.png"),
    )
    plot_feature_heatmap(
        encoded,
        labels,
        channel_names(encoding),
        Path(str(prefix) + ".feature_heatmap.png"),
    )
    _save_json(Path(str(prefix) + ".data_split.json"), {
        "all": {"samples": int(len(labels)), "positive": int(np.sum(labels == 1)), "negative": int(np.sum(labels == 0))},
        "training": {"samples": int(len(train_indices)), "positive": int(np.sum(labels[train_indices] == 1)), "negative": int(np.sum(labels[train_indices] == 0))},
        "validation": {"samples": int(len(validation_indices)), "positive": int(np.sum(y_validation == 1)), "negative": int(np.sum(y_validation == 0))},
    })

    weight_paths = []
    histories = []
    member_summaries = []
    ensemble_validation_scores = np.zeros(len(validation_indices), dtype=np.float64)
    for member_index in range(model_count):
        member_seed = seed + member_index
        relative_blocks, relative_dropped = sequential_balanced_blocks(
            labels[train_indices], member_seed, return_dropped=True
        )
        member_blocks = [train_indices[relative] for relative in relative_blocks]
        dropped_negative_indices = train_indices[relative_dropped]
        weight_path = _weight_path(prefix, member_index)
        member_prefix = _member_prefix(prefix, member_index)

        print(
            f"[base] member={member_index + 1}/{model_count}, "
            f"raw_train={len(train_indices)}, blocks={len(member_blocks)}, "
            f"validation={len(validation_indices)}, seed={member_seed}, "
            f"epochs_per_block={epochs}, max_total_epochs={len(member_blocks) * epochs}"
        )
        _write_training_blocks(
            Path(str(member_prefix) + ".training_blocks.tsv"),
            examples,
            member_blocks,
        )
        dropped_manifest_path = Path(
            str(member_prefix) + ".dropped_negative_remainder.tsv"
        )
        _write_dropped_negative_remainder(
            dropped_manifest_path,
            examples,
            dropped_negative_indices,
        )
        print(
            "[negative remainder] dropped={0}, manifest={1}".format(
                len(dropped_negative_indices), dropped_manifest_path
            )
        )
        model = build_network(
            encoded.shape[1], encoded.shape[2], learning_rate=learning_rate, seed=member_seed
        )
        if member_index == 0:
            save_model_description(model, prefix)
        result = _fit_member(
            model,
            encoded,
            labels,
            member_blocks,
            train_indices,
            x_validation,
            y_validation,
            weight_path,
            member_prefix,
            epochs,
            batch_size,
            patience,
        )
        _write_scored_examples(
            Path(str(member_prefix) + ".validation.predictions.tsv"),
            examples,
            validation_indices,
            result["validation_scores"],
        )
        ensemble_validation_scores += result["validation_scores"]
        histories.append(result["history"])
        member_summaries.append({
            "member": member_index,
            "seed": member_seed,
            "training_strategy": "deepcleave_sequential_1_to_1_full_blocks_drop_remainder",
            "block_count": result["block_count"],
            "block_sample_counts": result["block_sample_counts"],
            "dropped_negative_count": int(len(dropped_negative_indices)),
            "dropped_negative_manifest": dropped_manifest_path.name,
            "checkpoint": result["checkpoint_summary"],
            "history_summary": summarize_history(result["history"]),
            "training_metrics": result["training_metrics"],
            "validation_metrics": result["validation_metrics"],
        })
        weight_paths.append(weight_path)

    ensemble_validation_scores /= float(model_count)
    ensemble_metrics = save_evaluation_bundle(
        Path(str(prefix) + ".ensemble.validation"),
        y_validation,
        ensemble_validation_scores,
        threshold=0.5,
        title="Ensemble validation evaluation",
    )
    _write_scored_examples(
        Path(str(prefix) + ".ensemble.validation.predictions.tsv"),
        examples,
        validation_indices,
        ensemble_validation_scores,
    )
    plot_ensemble_history(histories, Path(str(prefix) + ".ensemble.training_curves.png"))

    config = _base_config("base", focus, radius, encoding, seed, weight_paths)
    config.update({
        "input_file": str(Path(input_path)),
        "protein_count": len(protein_ids),
        "candidate_count": len(examples),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "validation_fraction": validation_fraction,
        "split_unit": "protein",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_per_block": epochs,
        "training_strategy": "deepcleave_sequential_1_to_1_full_blocks_drop_remainder",
        "negative_remainder_policy": "drop_incomplete_block_like_deepcleave",
        "all_training_negatives_used_per_member": bool(
            all(item["dropped_negative_count"] == 0 for item in member_summaries)
        ),
        "checkpoint_strategy": "global_best_val_loss_across_all_blocks",
        "metric_threshold": 0.5,
        "member_summaries": member_summaries,
        "ensemble_validation_metrics": ensemble_metrics,
        "run_started_utc": run_started_utc,
        "run_finished_utc": _utc_now(),
        "run_seconds": float(time.time() - run_started_clock),
        "observed_runtime": _runtime_manifest(),
    })
    config["artifact_files"] = _artifact_files(prefix)
    _save_json(_config_path(prefix), config)
    return config


def train_transfer(
    input_path,
    background_prefix,
    output_prefix,
    focus_residues=None,
    radius=None,
    models_per_source=1,
    unfreeze_last=4,
    epochs=60,
    batch_size=256,
    validation_fraction=0.1,
    patience=10,
    learning_rate=2e-4,
    seed=42,
):

    if models_per_source < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("models, epochs, and batch-size must all be at least 1.")
    run_started_utc = _utc_now()
    run_started_clock = time.time()
    prefix = _prefix_path(output_prefix)
    background = load_config(background_prefix)
    source_weights = _resolve_weight_files(background_prefix, background)
    background_radius = int(background["radius"])
    if radius is None:
        radius = background_radius
    else:
        radius = int(radius)
        if radius != background_radius:
            raise ValueError(
                "train-transfer radius must match the background model: "
                f"specified {radius}, background model has {background_radius}."
            )
    encoding = normalize_encoding_name(background["encoding"])
    focus = parse_focus_residues(focus_residues or background["focus_residues"])

    examples, protein_ids = load_examples(input_path, focus, radius)
    labels = _labels(examples)
    train_indices, validation_indices = grouped_split(
        labels,
        [example.protein_id for example in examples],
        validation_fraction,
        seed,
    )
    encoded = encode_sequences([example.sequence for example in examples], encoding)
    if encoded.shape[2] != int(background["channel_count"]):
        raise ValueError("The background configuration channel_count is incompatible with the current encoder.")

    y_validation = labels[validation_indices]
    plot_data_overview(
        labels,
        labels[train_indices],
        y_validation,
        Path(str(prefix) + ".data_split.png"),
    )
    plot_feature_heatmap(
        encoded,
        labels,
        channel_names(encoding),
        Path(str(prefix) + ".feature_heatmap.png"),
    )
    _save_json(Path(str(prefix) + ".data_split.json"), {
        "all": {"samples": int(len(labels)), "positive": int(np.sum(labels == 1)), "negative": int(np.sum(labels == 0))},
        "training": {"samples": int(len(train_indices)), "positive": int(np.sum(labels[train_indices] == 1)), "negative": int(np.sum(labels[train_indices] == 0))},
        "validation": {"samples": int(len(validation_indices)), "positive": int(np.sum(y_validation == 1)), "negative": int(np.sum(y_validation == 0))},
    })

    weight_paths = []
    histories = []
    member_summaries = []
    total_members = len(source_weights) * models_per_source
    ensemble_validation_scores = np.zeros(len(validation_indices), dtype=np.float64)
    member_index = 0
    for source_index, source_weight in enumerate(source_weights):
        for repeat_index in range(models_per_source):
            member_seed = seed + member_index
            relative_blocks, relative_dropped = sequential_balanced_blocks(
                labels[train_indices], member_seed, return_dropped=True
            )
            member_blocks = [train_indices[relative] for relative in relative_blocks]
            dropped_negative_indices = train_indices[relative_dropped]
            weight_path = _weight_path(prefix, member_index)
            member_prefix = _member_prefix(prefix, member_index)
            print(
                f"[transfer] source={source_index + 1}/{len(source_weights)}, "
                f"repeat={repeat_index + 1}/{models_per_source}, raw_train={len(train_indices)}, "
                f"blocks={len(member_blocks)}, validation={len(validation_indices)}, seed={member_seed}, "
                f"epochs_per_block={epochs}, max_total_epochs={len(member_blocks) * epochs}"
            )
            _write_training_blocks(
                Path(str(member_prefix) + ".training_blocks.tsv"),
                examples,
                member_blocks,
            )
            dropped_manifest_path = Path(
                str(member_prefix) + ".dropped_negative_remainder.tsv"
            )
            _write_dropped_negative_remainder(
                dropped_manifest_path,
                examples,
                dropped_negative_indices,
            )
            print(
                "[negative remainder] dropped={0}, manifest={1}".format(
                    len(dropped_negative_indices), dropped_manifest_path
                )
            )
            model = initialize_transfer_network(
                sequence_length=encoded.shape[1],
                channel_count=encoded.shape[2],
                source_weights=str(source_weight),
                unfreeze_last=unfreeze_last,
                learning_rate=learning_rate,
                seed=member_seed,
            )
            if member_index == 0:
                save_model_description(model, prefix)
            result = _fit_member(
                model,
                encoded,
                labels,
                member_blocks,
                train_indices,
                encoded[validation_indices],
                labels[validation_indices],
                weight_path,
                member_prefix,
                epochs,
                batch_size,
                patience,
            )
            _write_scored_examples(
                Path(str(member_prefix) + ".validation.predictions.tsv"),
                examples,
                validation_indices,
                result["validation_scores"],
            )
            ensemble_validation_scores += result["validation_scores"]
            histories.append(result["history"])
            member_summaries.append({
                "member": member_index,
                "source_weight": source_weight.name,
                "seed": member_seed,
                "training_strategy": "deepcleave_sequential_1_to_1_full_blocks_drop_remainder",
                "block_count": result["block_count"],
                "block_sample_counts": result["block_sample_counts"],
                "dropped_negative_count": int(len(dropped_negative_indices)),
                "dropped_negative_manifest": dropped_manifest_path.name,
                "checkpoint": result["checkpoint_summary"],
                "history_summary": summarize_history(result["history"]),
                "training_metrics": result["training_metrics"],
                "validation_metrics": result["validation_metrics"],
            })
            weight_paths.append(weight_path)
            member_index += 1

    ensemble_validation_scores /= float(total_members)
    ensemble_metrics = save_evaluation_bundle(
        Path(str(prefix) + ".ensemble.validation"),
        y_validation,
        ensemble_validation_scores,
        threshold=0.5,
        title="Transfer ensemble validation evaluation",
    )
    _write_scored_examples(
        Path(str(prefix) + ".ensemble.validation.predictions.tsv"),
        examples,
        validation_indices,
        ensemble_validation_scores,
    )
    plot_ensemble_history(histories, Path(str(prefix) + ".ensemble.training_curves.png"))

    config = _base_config("transfer", focus, radius, encoding, seed, weight_paths)
    config.update({
        "input_file": str(Path(input_path)),
        "background_config": str(_config_path(background_prefix)),
        "background_model_count": len(source_weights),
        "models_per_source": models_per_source,
        "unfreeze_last": unfreeze_last,
        "protein_count": len(protein_ids),
        "candidate_count": len(examples),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "validation_fraction": validation_fraction,
        "split_unit": "protein",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_per_block": epochs,
        "training_strategy": "deepcleave_sequential_1_to_1_full_blocks_drop_remainder",
        "negative_remainder_policy": "drop_incomplete_block_like_deepcleave",
        "all_training_negatives_used_per_member": bool(
            all(item["dropped_negative_count"] == 0 for item in member_summaries)
        ),
        "checkpoint_strategy": "global_best_val_loss_across_all_blocks",
        "metric_threshold": 0.5,
        "member_summaries": member_summaries,
        "ensemble_validation_metrics": ensemble_metrics,
        "run_started_utc": run_started_utc,
        "run_finished_utc": _utc_now(),
        "run_seconds": float(time.time() - run_started_clock),
        "observed_runtime": _runtime_manifest(),
    })
    config["artifact_files"] = _artifact_files(prefix)
    _save_json(_config_path(prefix), config)
    return config


def _auc(labels, scores):

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if not positive_count or not negative_count:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def classification_metrics(labels, scores, threshold):

    return binary_metrics(labels, scores, threshold)


def predict(input_path, model_prefix, output_path, threshold=0.5, batch_size=512):

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    config = load_config(model_prefix)
    weights = _resolve_weight_files(model_prefix, config)
    examples, _ = load_examples(input_path, config["focus_residues"], int(config["radius"]))
    encoded = encode_sequences([example.sequence for example in examples], config["encoding"])
    if encoded.shape[2] != int(config["channel_count"]):
        raise ValueError("The model configuration channel_count is incompatible with the current encoder.")
    scores = np.zeros(len(examples), dtype=np.float64)

    for member_index, weight_path in enumerate(weights):
        print(f"[predict] member={member_index + 1}/{len(weights)}: {weight_path.name}")
        model = build_network(
            encoded.shape[1],
            encoded.shape[2],
            learning_rate=float(config.get("learning_rate", 1e-3)),
            seed=int(config.get("seed", 42)) + member_index,
        )
        model.load_weights(str(weight_path))
        scores += model.predict(encoded, batch_size=batch_size, verbose=0).reshape(-1)
    scores /= len(weights)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "protein_id", "position_0based", "position_1based", "residue",
            "window_sequence", "fasta_label", "probability", "prediction",
        ])
        for example, score in zip(examples, scores):
            writer.writerow([
                example.protein_id,
                example.position0,
                example.position1,
                example.residue,
                example.sequence,
                example.label,
                f"{score:.8f}",
                int(score >= threshold),
            ])

    labels = _labels(examples)
    metrics = None
    if len(np.unique(labels)) == 2:
        metrics = classification_metrics(labels, scores, threshold)
        evaluation_prefix = output.parent / (output.stem + ".evaluation")
        metrics = save_evaluation_bundle(
            evaluation_prefix,
            labels,
            scores,
            threshold=threshold,
            title="Prediction evaluation: " + output.name,
        )
    return scores, metrics


def inspect_dataset(input_path, focus_residues, radius, encoding, preview=5):

    focus = parse_focus_residues(focus_residues)
    examples, protein_ids = load_examples(input_path, focus, radius)
    labels = _labels(examples)
    encoded = encode_sequences([example.sequence for example in examples], encoding)
    summary = {
        "protein_count": len(protein_ids),
        "candidate_count": len(examples),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "tensor_shape": list(encoded.shape),
        "all_values_finite": bool(np.isfinite(encoded).all()),
        "minimum": float(encoded.min()),
        "maximum": float(encoded.max()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[preview]")
    for example in examples[: max(0, preview)]:
        print(
            f"label={example.label} id={example.protein_id!r} "
            f"position1={example.position1} residue={example.residue} window={example.sequence}"
        )
    return summary
