"""Reusable, privacy-safe components for hierarchical ticket classification."""

from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_classifier(c: float = 1.5) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(C=c, class_weight="balanced", max_iter=10_000)),
    ])


def explicit_type_rule(text: object) -> str | None:
    normalized = normalize_text(text)
    if normalized.startswith("complaint"):
        return "Complaint"
    if "provide support" in normalized:
        return "Support Request"
    return None


def predict_records(bundle: dict, titles: list[str]) -> list[dict[str, str]]:
    normalized = [normalize_text(title) for title in titles]
    services = bundle["service_model"].predict(normalized)
    model_types = bundle["type_model"].predict(normalized)
    rows = []
    for title, clean, service, model_type in zip(titles, normalized, services, model_types):
        ticket_type = explicit_type_rule(title) or model_type
        indicator_model = bundle["indicator_models"].get(service)
        indicator = (
            indicator_model.predict([clean])[0]
            if indicator_model is not None
            else bundle["indicator_fallbacks"].get(service, bundle["global_indicator_fallback"])
        )
        rows.append({
            "title": title,
            "predicted_service": service,
            "predicted_ticket_type": ticket_type,
            "predicted_indicator": indicator,
        })
    return rows
