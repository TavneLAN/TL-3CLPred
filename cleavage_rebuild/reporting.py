
import csv
import json
import time
from pathlib import Path

import numpy as np


HISTORY_COLUMNS = [
    "epoch",
    "block",
    "block_epoch",
    "block_train_samples",
    "loss",
    "acc",
    "mcc",
    "val_loss",
    "val_acc",
    "val_mcc",
    "learning_rate",
    "epoch_seconds",
    "elapsed_seconds",
]


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _finite_float(value):

    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _rank_auc(labels, scores):

    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.sum(truth == 1))
    negative_count = int(np.sum(truth == 0))
    if not positive_count or not negative_count:
        return None

    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while end < len(probabilities) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[truth == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_metrics(labels, scores, threshold=0.5):

    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != probabilities.shape:
        raise ValueError("labels and scores must have the same length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(np.int8)
    tp = int(np.sum((predictions == 1) & (truth == 1)))
    tn = int(np.sum((predictions == 0) & (truth == 0)))
    fp = int(np.sum((predictions == 1) & (truth == 0)))
    fn = int(np.sum((predictions == 0) & (truth == 1)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    npv = _safe_divide(tn, tn + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    mcc_denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return {
        "threshold": float(threshold),
        "accuracy": _safe_divide(tp + tn, len(truth)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1": f1,
        "mcc": _safe_divide(tp * tn - fp * fn, mcc_denominator),
        "roc_auc": _rank_auc(truth, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sample_count": int(len(truth)),
        "positive_count": int(np.sum(truth == 1)),
        "negative_count": int(np.sum(truth == 0)),
    }


def make_training_monitor(
    keras,
    x_train,
    y_train,
    x_validation,
    y_validation,
    batch_size,
    live_csv_path=None,
    live_json_path=None,
    shared_records=None,
    block_index=0,
    initial_epoch=0,
    elapsed_offset=0.0,
):

    class FullDatasetMetrics(keras.callbacks.Callback):
        def __init__(self):
            super(FullDatasetMetrics, self).__init__()
            self.records = shared_records if shared_records is not None else []
            self._training_started = None
            self._epoch_started = None

        def on_train_begin(self, logs=None):
            self._training_started = time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_started = time.time()

        def _learning_rate(self):
            try:
                return float(keras.backend.get_value(self.model.optimizer.lr))
            except Exception:
                return None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            train_scores = self.model.predict(
                x_train, batch_size=batch_size, verbose=0
            ).reshape(-1)
            validation_scores = self.model.predict(
                x_validation, batch_size=batch_size, verbose=0
            ).reshape(-1)
            train_metrics = binary_metrics(y_train, train_scores, threshold=0.5)
            validation_metrics = binary_metrics(y_validation, validation_scores, threshold=0.5)


            train_acc = logs.get("acc", logs.get("accuracy", train_metrics["accuracy"]))
            validation_acc = logs.get(
                "val_acc", logs.get("val_accuracy", validation_metrics["accuracy"])
            )



            logs["mcc"] = np.float32(train_metrics["mcc"])
            logs["val_mcc"] = np.float32(validation_metrics["mcc"])

            now = time.time()
            record = {
                "epoch": int(epoch + 1),
                "block": int(block_index + 1),
                "block_epoch": int(epoch - initial_epoch + 1),
                "block_train_samples": int(len(y_train)),
                "loss": _finite_float(logs.get("loss")),
                "acc": _finite_float(train_acc),
                "mcc": _finite_float(train_metrics["mcc"]),
                "val_loss": _finite_float(logs.get("val_loss")),
                "val_acc": _finite_float(validation_acc),
                "val_mcc": _finite_float(validation_metrics["mcc"]),
                "learning_rate": _finite_float(self._learning_rate()),
                "epoch_seconds": _finite_float(now - self._epoch_started),
                "elapsed_seconds": _finite_float(
                    float(elapsed_offset) + now - self._training_started
                ),
            }
            self.records.append(record)


            if live_csv_path is not None and live_json_path is not None:
                write_history(live_csv_path, live_json_path, self.records)
            print(
                "[epoch metrics] mcc={0:.5f}, val_mcc={1:.5f}, epoch_seconds={2:.2f}".format(
                    record["mcc"], record["val_mcc"], record["epoch_seconds"]
                )
            )

    return FullDatasetMetrics()


def summarize_history(records):

    if not records:
        return {"epochs_completed": 0}

    def best_record(key, minimize=False):
        candidates = [record for record in records if record.get(key) is not None]
        if not candidates:
            return None
        selector = min if minimize else max
        return selector(candidates, key=lambda item: item[key])

    best_loss = best_record("val_loss", minimize=True)
    best_acc = best_record("val_acc")
    best_mcc = best_record("val_mcc")
    return {
        "epochs_completed": int(len(records)),
        "total_seconds": records[-1].get("elapsed_seconds"),
        "best_val_loss": None if best_loss is None else best_loss["val_loss"],
        "best_val_loss_epoch": None if best_loss is None else best_loss["epoch"],
        "best_val_acc": None if best_acc is None else best_acc["val_acc"],
        "best_val_acc_epoch": None if best_acc is None else best_acc["epoch"],
        "best_val_mcc": None if best_mcc is None else best_mcc["val_mcc"],
        "best_val_mcc_epoch": None if best_mcc is None else best_mcc["epoch"],
    }


def write_history(csv_path, json_path, records):

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in HISTORY_COLUMNS})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(plt, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close()


def _history_values(records, key):
    return np.asarray([
        np.nan if record.get(key) is None else float(record[key]) for record in records
    ], dtype=np.float64)


def plot_training_history(records, path, title):

    if not records:
        return
    plt = _pyplot()
    epochs = _history_values(records, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    panels = [
        (axes[0, 0], "loss", "val_loss", "Loss"),
        (axes[0, 1], "acc", "val_acc", "Accuracy"),
        (axes[1, 0], "mcc", "val_mcc", "Matthews correlation coefficient"),
    ]
    for axis, train_key, validation_key, label in panels:
        axis.plot(epochs, _history_values(records, train_key), label="train", linewidth=2)
        axis.plot(epochs, _history_values(records, validation_key), label="validation", linewidth=2)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()


    block_boundaries = []
    previous_block = records[0].get("block")
    for record in records[1:]:
        current_block = record.get("block")
        if current_block is not None and current_block != previous_block:
            block_boundaries.append(float(record["epoch"]) - 0.5)
        previous_block = current_block
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        for boundary in block_boundaries:
            axis.axvline(boundary, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)

    time_axis = axes[1, 1]
    time_axis.bar(epochs, _history_values(records, "epoch_seconds"), alpha=0.65, label="epoch seconds")
    time_axis.set_xlabel("Epoch")
    time_axis.set_ylabel("Seconds")
    time_axis.grid(alpha=0.25)
    learning_axis = time_axis.twinx()
    learning_axis.plot(
        epochs, _history_values(records, "learning_rate"), color="#b33c86", marker=".", label="learning rate"
    )
    learning_axis.set_ylabel("Learning rate")
    for boundary in block_boundaries:
        time_axis.axvline(boundary, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    _finish_figure(plt, path)


def plot_ensemble_history(histories, path):

    histories = [records for records in histories if records]
    if not histories:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Ensemble training history (mean +/- one standard deviation)")
    for axis, train_key, validation_key, label in [
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "acc", "val_acc", "Accuracy"),
        (axes[2], "mcc", "val_mcc", "MCC"),
    ]:
        max_epochs = max(len(records) for records in histories)
        for key, color, curve_label in [
            (train_key, "#2878b5", "train"),
            (validation_key, "#d9534f", "validation"),
        ]:
            matrix = np.full((len(histories), max_epochs), np.nan, dtype=np.float64)
            for row, records in enumerate(histories):
                values = _history_values(records, key)
                matrix[row, :len(values)] = values
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            epochs = np.arange(1, max_epochs + 1)
            axis.plot(epochs, mean, color=color, label=curve_label, linewidth=2)
            axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    _finish_figure(plt, path)


def _curve_arrays(labels, scores):
    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    thresholds = np.linspace(1.0, 0.0, 201)
    rows = [binary_metrics(truth, probabilities, threshold) for threshold in thresholds]
    fpr = np.asarray([1.0 - row["specificity"] for row in rows])
    tpr = np.asarray([row["recall"] for row in rows])
    precision = np.asarray([row["precision"] for row in rows])
    return thresholds, rows, fpr, tpr, precision


def _precision_recall_arrays(labels, scores):

    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != probabilities.shape:
        raise ValueError("labels and scores must have the same length.")

    positive_count = int(np.sum(truth == 1))
    if not positive_count:
        return np.asarray([0.0]), np.asarray([0.0]), 0.0

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_truth = truth[order]
    sorted_scores = probabilities[order]
    true_positives = np.cumsum(sorted_truth == 1)
    false_positives = np.cumsum(sorted_truth == 0)


    distinct_ends = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_ends = np.concatenate([
        distinct_ends,
        np.asarray([len(sorted_scores) - 1], dtype=np.int64),
    ])
    tp = true_positives[threshold_ends].astype(np.float64)
    fp = false_positives[threshold_ends].astype(np.float64)
    recall = tp / float(positive_count)
    precision = tp / (tp + fp)


    recall = np.concatenate([np.asarray([0.0]), recall])
    precision = np.concatenate([np.asarray([1.0]), precision])
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    return recall, precision, average_precision


def _plot_confusion(axis, metrics):
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Predicted 0", "Predicted 1"])
    axis.set_yticklabels(["Actual 0", "Actual 1"])
    axis.set_title("Confusion matrix")
    return image


def _plot_roc(axis, fpr, tpr, auc_value):
    axis.plot(fpr, tpr, linewidth=2, label="ROC-AUC={0:.4f}".format(auc_value))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curve")
    axis.legend()
    axis.grid(alpha=0.25)


def _plot_pr(axis, recall, precision, positive_fraction, average_precision):
    axis.step(
        recall,
        precision,
        where="post",
        linewidth=2,
        label="AUPRC (AP)={0:.4f}".format(average_precision),
    )
    axis.axhline(positive_fraction, linestyle="--", color="grey", label="class baseline")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curve")
    axis.legend()
    axis.grid(alpha=0.25)
    return average_precision


def _plot_distribution(axis, labels, scores):
    truth = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(scores, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 31)
    axis.hist(probabilities[truth == 0], bins=bins, alpha=0.65, label="negative", density=True)
    axis.hist(probabilities[truth == 1], bins=bins, alpha=0.65, label="positive", density=True)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Density")
    axis.set_title("Prediction distributions")
    axis.legend()


def _plot_calibration(axis, labels, scores):
    truth = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(scores, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, 11)
    predicted_means = []
    observed_rates = []
    for index in range(10):
        if index == 9:
            selected = (probabilities >= boundaries[index]) & (probabilities <= boundaries[index + 1])
        else:
            selected = (probabilities >= boundaries[index]) & (probabilities < boundaries[index + 1])
        if np.any(selected):
            predicted_means.append(float(np.mean(probabilities[selected])))
            observed_rates.append(float(np.mean(truth[selected])))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    axis.plot(predicted_means, observed_rates, marker="o", linewidth=2, label="model")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive rate")
    axis.set_title("Calibration curve")
    axis.legend()
    axis.grid(alpha=0.25)


def _plot_thresholds(axis, thresholds, rows):
    for key, label in [
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("recall", "Sensitivity"),
        ("specificity", "Specificity"),
    ]:
        axis.plot(thresholds, [row[key] for row in rows], label=label)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Metric value")
    axis.set_title("Threshold analysis")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)


def save_evaluation_bundle(base_path, labels, scores, threshold=0.5, title="Validation evaluation"):

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = binary_metrics(labels, scores, threshold)
    thresholds, rows, fpr, tpr, _ = _curve_arrays(labels, scores)
    pr_recall, pr_precision, average_precision = _precision_recall_arrays(labels, scores)
    positive_fraction = _safe_divide(np.sum(np.asarray(labels) == 1), len(labels))

    metrics["pr_auc"] = average_precision
    metrics["average_precision"] = average_precision
    metrics["pr_auc_method"] = "average_precision"
    with Path(str(base_path) + ".metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    plot_specs = [
        ("confusion_matrix", lambda axis: _plot_confusion(axis, metrics)),
        ("roc_curve", lambda axis: _plot_roc(axis, fpr, tpr, metrics["roc_auc"])),
        (
            "precision_recall",
            lambda axis: _plot_pr(
                axis, pr_recall, pr_precision, positive_fraction, average_precision
            ),
        ),
        ("score_distribution", lambda axis: _plot_distribution(axis, labels, scores)),
        ("calibration", lambda axis: _plot_calibration(axis, labels, scores)),
        ("threshold_analysis", lambda axis: _plot_thresholds(axis, thresholds, rows)),
    ]
    for suffix, draw in plot_specs:
        plt = _pyplot()
        plt.figure(figsize=(6.2, 5.2))
        draw(plt.gca())
        _finish_figure(plt, Path(str(base_path) + "." + suffix + ".png"))

    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(title)
    _plot_confusion(axes[0, 0], metrics)
    _plot_roc(axes[0, 1], fpr, tpr, metrics["roc_auc"])
    _plot_pr(
        axes[0, 2], pr_recall, pr_precision, positive_fraction, average_precision
    )
    _plot_distribution(axes[1, 0], labels, scores)
    _plot_calibration(axes[1, 1], labels, scores)
    _plot_thresholds(axes[1, 2], thresholds, rows)
    _finish_figure(plt, Path(str(base_path) + ".dashboard.png"))
    return metrics


def save_model_description(model, prefix):

    prefix = Path(prefix)
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    Path(str(prefix) + ".model.summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    Path(str(prefix) + ".model.architecture.json").write_text(
        model.to_json(indent=2), encoding="utf-8"
    )

    layer_rows = []
    for index, layer in enumerate(model.layers):
        try:
            output_shape = str(layer.output_shape)
        except Exception:
            output_shape = "unknown"
        layer_rows.append({
            "index": index,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": output_shape,
            "parameters": int(layer.count_params()),
            "trainable": bool(layer.trainable),
        })
    with Path(str(prefix) + ".model.layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "name", "type", "output_shape", "parameters", "trainable"],
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    plt = _pyplot()
    height = max(8.0, 0.38 * len(layer_rows) + 1.5)
    fig, axis = plt.subplots(figsize=(13, height))
    colors = {
        "InputLayer": "#d9edf7",
        "Conv1D": "#5bc0de",
        "BatchNormalization": "#bce8f1",
        "Activation": "#5cb85c",
        "Bidirectional": "#f0ad4e",
        "CuDNNLSTM": "#f0ad4e",
        "LSTM": "#f0ad4e",
        "Dense": "#d9534f",
        "Dropout": "#999999",
        "SpatialDropout1D": "#999999",
        "Lambda": "#9467bd",
    }
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.5, len(layer_rows) - 0.5)
    axis.axis("off")
    for row, layer in enumerate(reversed(layer_rows)):
        y_value = row
        color = colors.get(layer["type"], "#eeeeee")
        axis.add_patch(plt.Rectangle((0.02, y_value - 0.35), 0.96, 0.7, color=color, ec="#555555"))
        label = "{0:02d}  {1}  [{2}]  output={3}  params={4:,}  trainable={5}".format(
            layer["index"], layer["name"], layer["type"], layer["output_shape"],
            layer["parameters"], layer["trainable"],
        )
        axis.text(0.04, y_value, label, va="center", fontsize=8)
    axis.set_title("Model architecture: {0:,} parameters".format(model.count_params()), pad=14)
    _finish_figure(plt, Path(str(prefix) + ".model.architecture.png"))
    return layer_rows


def plot_data_overview(all_labels, train_labels, validation_labels, path):

    plt = _pyplot()
    arrays = [np.asarray(all_labels), np.asarray(train_labels), np.asarray(validation_labels)]
    negative = [int(np.sum(values == 0)) for values in arrays]
    positive = [int(np.sum(values == 1)) for values in arrays]
    x_values = np.arange(3)
    plt.figure(figsize=(8, 5))
    plt.bar(x_values - 0.18, negative, width=0.36, label="negative")
    plt.bar(x_values + 0.18, positive, width=0.36, label="positive")
    plt.xticks(x_values, ["All candidates", "Training split", "Validation split"])
    plt.ylabel("Sample count")
    plt.title("Dataset split and class distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _finish_figure(plt, path)


def plot_feature_heatmap(encoded, labels, channel_names, path):

    encoded = np.asarray(encoded, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int8)
    if not np.any(truth == 0) or not np.any(truth == 1):
        return
    negative_mean = np.mean(encoded[truth == 0], axis=0).T
    positive_mean = np.mean(encoded[truth == 1], axis=0).T
    difference = positive_mean - negative_mean
    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    matrices = [negative_mean, positive_mean, difference]
    titles = ["Negative-class mean", "Positive-class mean", "Positive minus negative"]
    for axis, matrix, title in zip(axes, matrices, titles):
        limit = max(1e-6, float(np.max(np.abs(matrix))))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_ylabel("Feature channel")
        axis.set_yticks(np.arange(len(channel_names)))
        axis.set_yticklabels(channel_names, fontsize=5)
        fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("Sequence-window position")
    _finish_figure(plt, path)


def plot_attention(attention_weights, labels, path):

    weights = np.asarray(attention_weights, dtype=np.float64)
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights[:, :, 0]
    truth = np.asarray(labels, dtype=np.int8)
    if weights.ndim != 2 or len(weights) != len(truth):
        return
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    positions = np.arange(weights.shape[1])
    for class_value, label, color in [(0, "negative", "#2878b5"), (1, "positive", "#d9534f")]:
        selected = weights[truth == class_value]
        if len(selected):
            axes[0].plot(positions, np.mean(selected, axis=0), label=label, color=color, linewidth=2)
    axes[0].set_title("Mean temporal attention by class")
    axes[0].set_xlabel("Downsampled sequence position")
    axes[0].set_ylabel("Attention weight")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    positive = weights[truth == 1]
    display = positive[: min(40, len(positive))] if len(positive) else weights[: min(40, len(weights))]
    image = axes[1].imshow(display, aspect="auto", cmap="viridis")
    axes[1].set_title("Attention heatmap (up to 40 validation samples)")
    axes[1].set_xlabel("Downsampled sequence position")
    axes[1].set_ylabel("Sample")
    fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.02)
    _finish_figure(plt, path)
