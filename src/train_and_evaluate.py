"""Train, evaluate and save the privacy-safe hierarchical classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from modeling import build_classifier, explicit_type_rule, normalize_text, predict_records

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "synthetic_tickets.csv"
RESULTS = ROOT / "results"
MODEL_FILE = ROOT / "models" / "ticket_classifier.joblib"
REQUIRED = {"title", "service", "ticket_type", "indicator"}


def evaluate(truth, predicted) -> dict:
    labels = sorted(set(truth) | set(predicted))
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(truth, predicted, average="weighted", zero_division=0)), 4),
        "labels": labels,
        "classification_report": classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0),
    }


def save_matrix(truth, predicted, labels, filename, title) -> None:
    matrix = confusion_matrix(truth, predicted, labels=labels)
    size = max(6, min(12, len(labels) * 0.75))
    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(RESULTS / filename, dpi=180); plt.close()


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run src/generate_synthetic_data.py first.")
    data = pd.read_csv(DATA_FILE)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Required training fields contain missing values.")
    data = data.drop_duplicates(subset=["title", "service", "ticket_type", "indicator"]).copy()
    data["normalized_title"] = data["title"].map(normalize_text)
    if data["normalized_title"].eq("").any():
        raise ValueError("Empty normalized title.")
    stratify = data["service"].astype(str) + "|" + data["ticket_type"].astype(str)
    if stratify.value_counts().min() < 2:
        raise ValueError("Each service/type stratum needs at least two rows.")
    train, test = train_test_split(data, test_size=.20, random_state=42, stratify=stratify)

    service_model = build_classifier().fit(train["normalized_title"], train["service"])
    type_model = build_classifier().fit(train["normalized_title"], train["ticket_type"])
    indicator_models, indicator_fallbacks = {}, {}
    global_fallback = train["indicator"].mode().iat[0]
    for service, group in train.groupby("service"):
        indicator_fallbacks[service] = group["indicator"].mode().iat[0]
        if group["indicator"].nunique() > 1 and len(group) >= 10:
            indicator_models[service] = build_classifier(c=2.0).fit(
                group["normalized_title"], group["indicator"])
    bundle = {
        "service_model": service_model,
        "type_model": type_model,
        "indicator_models": indicator_models,
        "indicator_fallbacks": indicator_fallbacks,
        "global_indicator_fallback": global_fallback,
        "schema_version": 1,
    }
    predicted = pd.DataFrame(predict_records(bundle, test["title"].astype(str).tolist()),
                             index=test.index)
    service_pred = predicted["predicted_service"]
    type_model_pred = type_model.predict(test["normalized_title"])
    type_final_pred = predicted["predicted_ticket_type"]
    indicator_pred = predicted["predicted_indicator"]
    metrics = {
        "data_origin": "deterministic synthetic generator",
        "rows": int(len(data)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split": "stratified random 80/20 after exact labeled-row deduplication",
        "caveat": "Random validation is not a temporal or grouped production estimate.",
        "service": evaluate(test["service"], service_pred),
        "ticket_type_model": evaluate(test["ticket_type"], type_model_pred),
        "ticket_type_final": evaluate(test["ticket_type"], type_final_pred),
        "indicator_hierarchical": evaluate(test["indicator"], indicator_pred),
        "indicator_models_trained": len(indicator_models),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    (RESULTS / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_matrix(test["service"], service_pred, metrics["service"]["labels"],
                "service_confusion_matrix.png", "Synthetic test set: service")
    save_matrix(test["ticket_type"], type_final_pred, metrics["ticket_type_final"]["labels"],
                "ticket_type_confusion_matrix.png", "Synthetic test set: ticket type")
    save_matrix(test["indicator"], indicator_pred, metrics["indicator_hierarchical"]["labels"],
                "indicator_confusion_matrix.png", "Synthetic test set: hierarchical indicator")
    print(json.dumps({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                      for k, v in metrics.items() if isinstance(v, dict) and "accuracy" in v}, indent=2))
    print(f"Saved model to {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
