"""CLI for validating, aggregating, and splitting generated datasets."""

from __future__ import annotations

import argparse

from .assembly import DatasetAssemblyError, assemble_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble generated JSONL samples into deterministic train/validation/test splits."
    )
    parser.add_argument("--input-glob", required=True, help="Glob for generated sample JSONL files.")
    parser.add_argument("--policy", default="dataset_policy.json", help="Dataset policy JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory for final split files.")
    parser.add_argument(
        "--allow-skipped-labels",
        action="store_true",
        help="Allow records whose label method is not exact_all_terminal.",
    )
    args = parser.parse_args(argv)

    try:
        result = assemble_dataset(
            input_glob=args.input_glob,
            output_dir=args.output_dir,
            policy_path=args.policy,
            allow_skipped_labels=args.allow_skipped_labels,
        )
    except DatasetAssemblyError as error:
        parser.error(str(error))

    print(
        "assembled dataset "
        f"samples={result.sample_count} matrix={result.matrix_size} "
        f"train={result.train_count} validation={result.validation_count} test={result.test_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
