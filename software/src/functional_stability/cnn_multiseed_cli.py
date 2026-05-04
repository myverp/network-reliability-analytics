"""CLI for stronger CNN multi-seed evaluation."""

from __future__ import annotations

import argparse

from .cnn_smoke import run_stronger_cnn_multiseed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate stronger_cnn across fixed random seeds on the assembled dataset."
    )
    parser.add_argument("--train", required=True, help="Train split JSONL path.")
    parser.add_argument("--validation", required=True, help="Validation split JSONL path.")
    parser.add_argument("--test", required=True, help="Test split JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated integer seeds.")
    parser.add_argument("--matrix-size", type=int, default=10, help="Expected adjacency matrix size.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs per seed.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop a seed after this many epochs without validation MAE improvement.",
    )
    parser.add_argument(
        "--ridge-test-mae",
        type=float,
        default=0.035949,
        help="Reference ridge test MAE for comparison.",
    )
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    result = run_stronger_cnn_multiseed(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_path=args.output,
        seeds=seeds,
        matrix_size=args.matrix_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        ridge_test_mae_reference=args.ridge_test_mae,
    )

    print("stronger_cnn multi-seed results")
    for row in result["per_seed"]:
        validation = row["metrics"]["validation"]
        test = row["metrics"]["test"]
        print(
            f"seed={row['seed']} "
            f"validation_mae={validation['mae']:.6g} "
            f"validation_rmse={validation['rmse']:.6g} "
            f"test_mae={test['mae']:.6g} "
            f"test_rmse={test['rmse']:.6g} "
            f"beats_ridge={row['beats_ridge_test_mae']}"
        )

    summary = result["summary"]
    print(
        "summary "
        f"validation_mae_mean={summary['validation']['mae']['mean']:.6g} "
        f"validation_mae_std={summary['validation']['mae']['std']:.6g} "
        f"validation_rmse_mean={summary['validation']['rmse']['mean']:.6g} "
        f"validation_rmse_std={summary['validation']['rmse']['std']:.6g} "
        f"test_mae_mean={summary['test']['mae']['mean']:.6g} "
        f"test_mae_std={summary['test']['mae']['std']:.6g} "
        f"test_rmse_mean={summary['test']['rmse']['mean']:.6g} "
        f"test_rmse_std={summary['test']['rmse']['std']:.6g} "
        f"consistently_beats_ridge={result['consistently_beats_ridge_test_mae']}"
    )
    return 0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
