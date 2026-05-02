"""CLI for simple non-CNN baseline evaluation."""

from __future__ import annotations

import argparse

from .baseline import run_baselines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate mean and ridge baselines on assembled dataset splits."
    )
    parser.add_argument("--train", required=True, help="Train split JSONL path.")
    parser.add_argument("--validation", required=True, help="Validation split JSONL path.")
    parser.add_argument("--test", required=True, help="Test split JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength.")
    args = parser.parse_args(argv)

    result = run_baselines(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_path=args.output,
        ridge_alpha=args.ridge_alpha,
    )
    print(
        "baseline results "
        f"mean_test_mae={result['models']['mean']['test']['mae']:.6g} "
        f"ridge_test_mae={result['models']['ridge']['test']['mae']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
