"""Train and evaluate privacy-safe hierarchical text classifiers."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "synthetic_tickets.csv"
RESULTS = ROOT / "results"


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_classifier() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ]
    )
    return Pipeline(
        [("features", features), ("classifier", LinearSVC(C=1.5, class_weight="balanced"))]
    )


def explicit_type_rule(text: str) -> str | None:
    normalized = normalize_text(text)
    if normalized.startswith("complaint"):
        return "Complaint"
    if "provide support" in normalized:
        return "Support Request"
    return None


def save_matrix(y_true, y_pred, labels: list[str], filename: str, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(6, min(12, len(labels) * 0.75))
    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=180)
    plt.close()


def evaluate(name: str, truth, predicted) -> dict:
    labels = sorted(set(truth) | set(predicted))
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "labels": labels,
        "classification_report": classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run src/generate_synthetic_data.py first.")

    data = pd.read_csv(DATA_FILE).drop_duplicates(subset=["title", "service", "ticket_type"])
    data["normalized_title"] = data["title"].map(normalize_text)
    stratify_key = data["service"].astype(str) + "|" + data["ticket_type"].astype(str)
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=stratify_key
    )

    service_model = build_classifier().fit(train["normalized_title"], train["service"])
    type_model = build_classifier().fit(train["normalized_title"], train["ticket_type"])
    indicator_model = build_classifier().fit(train["normalized_title"], train["indicator"])

    service_pred = service_model.predict(test["normalized_title"])
    type_model_pred = type_model.predict(test["normalized_title"])
    type_final_pred = [explicit_type_rule(text) or pred for text, pred in zip(test["title"], type_model_pred)]
    indicator_pred = indicator_model.predict(test["normalized_title"])

    metrics = {
        "data_origin": "deterministic synthetic generator",
        "rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split": "stratified random 80/20 holdout after exact-title deduplication",
        "service": evaluate("service", test["service"], service_pred),
        "ticket_type_model": evaluate("ticket_type_model", test["ticket_type"], type_model_pred),
        "ticket_type_final": evaluate("ticket_type_final", test["ticket_type"], type_final_pred),
        "indicator": evaluate("indicator", test["indicator"], indicator_pred),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_matrix(test["service"], service_pred, metrics["service"]["labels"], "service_confusion_matrix.png", "Synthetic test set: service")
    save_matrix(test["ticket_type"], type_final_pred, metrics["ticket_type_final"]["labels"], "ticket_type_confusion_matrix.png", "Synthetic test set: ticket type")
    save_matrix(test["indicator"], indicator_pred, metrics["indicator"]["labels"], "indicator_confusion_matrix.png", "Synthetic test set: indicator")

    summary = {key: value["accuracy"] for key, value in metrics.items() if isinstance(value, dict) and "accuracy" in value}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

