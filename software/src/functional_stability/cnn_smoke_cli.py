"""CLI for the minimal CNN smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cnn_smoke import run_cnn_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a small PyTorch CNN on 10x10 reliability matrices."
    )
    parser.add_argument("--train", required=True, help="Train split JSONL path.")
    parser.add_argument("--validation", required=True, help="Validation split JSONL path.")
    parser.add_argument("--test", required=True, help="Test split JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--matrix-size", type=int, default=10, help="Expected adjacency matrix size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--ridge-test-mae",
        type=float,
        default=0.035949,
        help="Reference ridge test MAE for comparison.",
    )
    args = parser.parse_args(argv)

    try:
        result = run_cnn_smoke(
            train_path=args.train,
            validation_path=args.validation,
            test_path=args.test,
            output_path=args.output,
            matrix_size=args.matrix_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            ridge_test_mae_reference=args.ridge_test_mae,
        )
    except RuntimeError as error:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "experiment": "cnn_smoke",
                    "status": "not_run",
                    "reason": str(error),
                    "required_dependency": "torch",
                    "install_command": "python -m pip install torch",
                    "ridge_test_mae_reference": args.ridge_test_mae,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        parser.exit(2, f"{error}\n")

    test = result["metrics"]["test"]
    validation = result["metrics"]["validation"]
    comparison = result["baseline_comparison"]
    print(
        "cnn smoke results "
        f"validation_mae={validation['mae']:.6g} "
        f"validation_rmse={validation['rmse']:.6g} "
        f"test_mae={test['mae']:.6g} "
        f"test_rmse={test['rmse']:.6g} "
        f"beats_ridge={comparison['beats_ridge_test_mae']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
