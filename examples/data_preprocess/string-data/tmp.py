#!/usr/bin/env python3
"""Keep the first N rows from a parquet file."""

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a parquet file and save only the first N rows."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output parquet file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of rows to keep from the top of the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 0:
        raise ValueError("--n must be >= 0")

    df = pd.read_parquet(args.input)
    df.head(args.n).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
