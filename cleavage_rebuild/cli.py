
import argparse
import json

from .features import channel_names, normalize_encoding_name
from .network import build_network
from .workflow import inspect_dataset, predict, train_base, train_transfer


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be an integer greater than or equal to 1.")
    return parsed


def _add_fit_arguments(parser, default_epochs, default_patience):
    parser.add_argument("--models", type=_positive_int, default=1, help="Number of ensemble members to train for each source model.")
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=default_epochs,
        help="Maximum number of training epochs for each DeepCleave-style 1:1 block.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=256, help="Mini-batch size.")
    parser.add_argument(
        "--validation-fraction", type=float, default=0.1, help="Validation protein fraction; must be less than 0.5."
    )
    parser.add_argument("--patience", type=int, default=default_patience, help="Early-stopping patience in epochs; 0 disables early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting and model initialization.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Independent reimplementation for protease cleavage site prediction: data inspection, training, transfer learning, and prediction."
    )

    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect FASTA records, labels, and feature tensors only.")
    inspect_parser.add_argument("--input", required=True, help="FASTA file with # positive-site markers.")
    inspect_parser.add_argument("--focus", required=True, help="Candidate center residues, for example Q or Q,E.")
    inspect_parser.add_argument("--radius", type=_positive_int, default=15)
    inspect_parser.add_argument("--encoding", default="physchem-v1", choices=["onehot", "physchem-v1"])
    inspect_parser.add_argument("--preview", type=int, default=5, help="Show the first candidate windows.")

    base_parser = subparsers.add_parser("train-base", help="Train the background/family-level model ensemble.")
    base_parser.add_argument("--input", required=True)
    base_parser.add_argument("--output-prefix", required=True)
    base_parser.add_argument("--focus", required=True)
    base_parser.add_argument("--radius", type=_positive_int, default=15)
    base_parser.add_argument("--encoding", default="physchem-v1", choices=["onehot", "physchem-v1"])
    base_parser.add_argument("--learning-rate", type=float, default=1e-3)
    _add_fit_arguments(base_parser, default_epochs=100, default_patience=15)

    transfer_parser = subparsers.add_parser("train-transfer", help="Run transfer learning from a background model.")
    transfer_parser.add_argument("--input", required=True)
    transfer_parser.add_argument("--background-prefix", required=True)
    transfer_parser.add_argument("--output-prefix", required=True)
    transfer_parser.add_argument(
        "--focus", default=None, help="Candidate residues for the target data; omit to reuse the background model setting."
    )
    transfer_parser.add_argument(
        "--radius",
        type=_positive_int,
        default=None,
        help="Window radius for target data; must match the background model setting. Omit to reuse the background model value.",
    )
    transfer_parser.add_argument(
        "--unfreeze-last", type=_positive_int, default=4, help="Unfreeze the last N layers that have weights."
    )
    transfer_parser.add_argument("--learning-rate", type=float, default=2e-4)
    _add_fit_arguments(transfer_parser, default_epochs=60, default_patience=10)

    predict_parser = subparsers.add_parser("predict", help="Average ensemble model probabilities and write a TSV file.")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--model-prefix", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--batch-size", type=_positive_int, default=512)

    smoke_parser = subparsers.add_parser("smoke-test", help="Build the model and show parameter count without training.")
    smoke_parser.add_argument("--radius", type=_positive_int, default=15)
    smoke_parser.add_argument("--encoding", default="physchem-v1", choices=["onehot", "physchem-v1"])
    smoke_parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_fit_options(args):
    if args.patience < 0:
        raise ValueError("patience cannot be less than 0.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be greater than 0.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("Specify inspect, train-base, train-transfer, predict, or smoke-test.")

    if args.command == "inspect":
        inspect_dataset(args.input, args.focus, args.radius, args.encoding, args.preview)
        return

    if args.command == "train-base":
        _validate_fit_options(args)
        config = train_base(
            input_path=args.input,
            output_prefix=args.output_prefix,
            focus_residues=args.focus,
            radius=args.radius,
            encoding=args.encoding,
            model_count=args.models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            patience=args.patience,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print("\n[Done] Model configuration:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    if args.command == "train-transfer":
        _validate_fit_options(args)
        config = train_transfer(
            input_path=args.input,
            background_prefix=args.background_prefix,
            output_prefix=args.output_prefix,
            focus_residues=args.focus,
            radius=args.radius,
            models_per_source=args.models,
            unfreeze_last=args.unfreeze_last,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            patience=args.patience,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print("\n[Done] Model configuration:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    if args.command == "predict":
        _, metrics = predict(
            input_path=args.input,
            model_prefix=args.model_prefix,
            output_path=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        print(f"\n[Done] Predictions written to: {args.output}")
        if metrics is not None:
            print("[evaluation]")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        else:
            print("The input does not contain both positive and negative labels, so classification metrics were not computed.")
        return

    if args.command == "smoke-test":
        encoding = normalize_encoding_name(args.encoding)
        model = build_network(
            sequence_length=2 * args.radius,
            channel_count=len(channel_names(encoding)),
            seed=args.seed,
        )
        model.summary()
        print(f"\n[Done] architecture={model.name}, parameters={model.count_params():,}")
