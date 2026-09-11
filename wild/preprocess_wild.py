"""
Converts wild trace CSVs into the format expected by TraceRequestGenerator:

  Date, Time, PromptTokenCount, CompletionTokenCount

Supported input column layouts (auto-detected):
  - TIMESTAMP, ContextTokens, GeneratedTokens        (conv_distributions_original.csv)
  - TIMESTAMP, num_prefill_tokens, num_decode_tokens  (code_distributions_original.csv)

Output columns:
  - Date  : calendar date extracted from TIMESTAMP (e.g. "2023-11-16")
  - Time  : seconds elapsed since the first request in the trace
  - PromptTokenCount     : prefill token count
  - CompletionTokenCount : decode / generated token count

Usage (process both files at once):
  python wild/preprocess_wild.py

Or specify individual files:
  python wild/preprocess_wild.py --input wild/conv_distributions_original.csv \
                                  --output wild/conv_distributions_processed.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Maps known column name variants -> canonical names
PREFILL_COLUMN_CANDIDATES = ["ContextTokens", "num_prefill_tokens"]
DECODE_COLUMN_CANDIDATES  = ["GeneratedTokens", "num_decode_tokens"]


def detect_token_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (prefill_col, decode_col) by inspecting the dataframe columns."""
    prefill_col = next((c for c in PREFILL_COLUMN_CANDIDATES if c in df.columns), None)
    decode_col  = next((c for c in DECODE_COLUMN_CANDIDATES  if c in df.columns), None)

    if prefill_col is None or decode_col is None:
        raise ValueError(
            f"Could not identify token columns. Found columns: {list(df.columns)}\n"
            f"Expected one of {PREFILL_COLUMN_CANDIDATES} for prefill and "
            f"one of {DECODE_COLUMN_CANDIDATES} for decode."
        )
    return prefill_col, decode_col


def preprocess(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    prefill_col, decode_col = detect_token_columns(df)
    print(f"  Detected columns: prefill='{prefill_col}', decode='{decode_col}'")

    # Parse timestamps
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Extract date string for the Date filter column
    df["Date"] = df["TIMESTAMP"].dt.strftime("%Y-%m-%d")

    # Convert timestamp to seconds elapsed since the first request
    t0 = df["TIMESTAMP"].min()
    df["Time"] = (df["TIMESTAMP"] - t0).dt.total_seconds()

    # Rename token columns to canonical names
    df = df.rename(columns={
        prefill_col: "PromptTokenCount",
        decode_col:  "CompletionTokenCount",
    })

    # Keep only the columns TraceRequestGenerator needs, sorted by arrival time
    out_df = (
        df[["Date", "Time", "PromptTokenCount", "CompletionTokenCount"]]
        .sort_values("Time")
        .reset_index(drop=True)
    )

    out_df.to_csv(output_path, index=False)

    print(f"  Written {len(out_df)} rows -> {output_path}")
    print(f"  Unique dates : {out_df['Date'].unique().tolist()}")
    print(f"  Time range   : {out_df['Time'].min():.2f}s – {out_df['Time'].max():.2f}s")
    print(f"  PromptTokenCount      — mean={out_df['PromptTokenCount'].mean():.1f}, "
          f"p50={out_df['PromptTokenCount'].median():.1f}, "
          f"max={out_df['PromptTokenCount'].max()}")
    print(f"  CompletionTokenCount  — mean={out_df['CompletionTokenCount'].mean():.1f}, "
          f"p50={out_df['CompletionTokenCount'].median():.1f}, "
          f"max={out_df['CompletionTokenCount'].max()}")


# Default pairs: (input, output)
DEFAULT_PAIRS = [
    ("wild/conv_distributions_original.csv", "wild/conv_distributions_processed.csv"),
    ("wild/code_distributions_original.csv", "wild/code_distributions_processed.csv"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess wild trace CSVs for TraceRequestGenerator."
    )
    parser.add_argument("--input",  default=None, help="Single input CSV path.")
    parser.add_argument("--output", default=None, help="Single output CSV path.")
    args = parser.parse_args()

    if args.input and args.output:
        pairs = [(args.input, args.output)]
    elif args.input or args.output:
        print("ERROR: provide both --input and --output, or neither (to process all defaults).")
        sys.exit(1)
    else:
        pairs = DEFAULT_PAIRS

    for inp, out in pairs:
        if not Path(inp).exists():
            print(f"[SKIP] {inp} not found.")
            continue
        print(f"\n[Processing] {inp}")
        preprocess(inp, out)

    print("\nDone. Use date='2023-11-16' in TraceRequestGeneratorConfig.")
