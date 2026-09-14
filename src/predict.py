"""Classify current authorized tickets with a previously trained public model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from modeling import predict_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify ticket titles from CSV.")
    parser.add_argument("input", type=Path, help="CSV containing a title column")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "predictions.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ticket_classifier.joblib")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if "title" not in data.columns:
        raise ValueError("Input CSV must contain a title column.")
    if data["title"].isna().any():
        raise ValueError("Title contains missing values.")
    bundle = joblib.load(args.model)
    predictions = pd.DataFrame(predict_records(bundle, data["title"].astype(str).tolist()))
    output = pd.concat([data.reset_index(drop=True), predictions.drop(columns="title")], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
