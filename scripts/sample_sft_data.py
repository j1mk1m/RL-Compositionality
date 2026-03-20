import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample N rows from a parquet file.")
    parser.add_argument("--input", required=True, help="Path to the input parquet file.")
    parser.add_argument("--output", required=True, help="Path to the output parquet file.")
    parser.add_argument("--n", type=int, required=True, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    if args.n > len(df):
        raise ValueError(f"Requested {args.n} samples but the dataset only has {len(df)} rows.")

    sampled = df.sample(n=args.n, random_state=args.seed)
    sampled.to_parquet(args.output, index=False)
    print(f"Saved {args.n} samples to {args.output}")


if __name__ == "__main__":
    main()
