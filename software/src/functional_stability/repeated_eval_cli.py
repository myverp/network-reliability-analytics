"""CLI for thesis-quality repeated stronger CNN vs ridge evaluation."""

from __future__ import annotations

import argparse

from .repeated_eval import run_repeated_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated stronger_cnn and ridge evaluation with confidence intervals."
    )
    parser.add_argument("--train", required=True, help="Train split JSONL path.")
    parser.add_argument("--validation", required=True, help="Validation split JSONL path.")
    parser.add_argument("--test", required=True, help="Test split JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--matrix-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-test-mae", type=float, default=0.035949)
    args = parser.parse_args(argv)

    result = run_repeated_evaluation(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_path=args.output,
        seeds=_parse_seeds(args.seeds),
        matrix_size=args.matrix_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        ridge_alpha=args.ridge_alpha,
        ridge_test_mae_reference=args.ridge_test_mae,
    )

    print("repeated stronger_cnn vs ridge")
    _print_model_summary("stronger_cnn", result["summary"]["stronger_cnn"]["overall"])
    _print_model_summary("ridge", result["summary"]["ridge"]["overall"])
    comparison = result["comparison"]
    print(
        "comparison "
        f"test_mae_difference_cnn_minus_ridge={comparison['test_mae_difference_cnn_minus_ridge']:.6g} "
        f"cnn_mean_test_mae_beats_ridge={comparison['cnn_mean_test_mae_beats_ridge']}"
    )
    return 0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _print_model_summary(model_name: str, summary: dict[str, object]) -> None:
    for split in ("validation", "test"):
        mae = summary[split]["mae"]
        rmse = summary[split]["rmse"]
        print(
            f"{model_name} {split} "
            f"mae_mean={mae['mean']:.6g} mae_std={mae['std']:.6g} "
            f"mae_ci95=[{mae['ci95_low']:.6g},{mae['ci95_high']:.6g}] "
            f"rmse_mean={rmse['mean']:.6g} rmse_std={rmse['std']:.6g} "
            f"rmse_ci95=[{rmse['ci95_low']:.6g},{rmse['ci95_high']:.6g}]"
        )


if __name__ == "__main__":
    raise SystemExit(main())
