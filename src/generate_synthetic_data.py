"""Generate fictional operational tickets for the public NLP demonstration."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic_tickets.csv"
SEED = 42

SERVICES = {
    "Facilities": {
        "objects": ["air conditioner", "water leak", "door", "lighting", "elevator"],
        "actions": ["inspect", "repair", "adjust", "restore", "check"],
        "indicator": "IND-01",
    },
    "IT Support": {
        "objects": ["network", "printer", "user account", "laptop", "application"],
        "actions": ["reset", "connect", "configure", "diagnose", "update"],
        "indicator": "IND-02",
    },
    "Logistics": {
        "objects": ["package", "delivery cart", "shipment", "document box", "supply order"],
        "actions": ["deliver", "collect", "transfer", "locate", "schedule"],
        "indicator": "IND-03",
    },
    "Food Services": {
        "objects": ["meal order", "beverage station", "menu", "food tray", "special meal"],
        "actions": ["prepare", "replace", "deliver", "review", "confirm"],
        "indicator": "IND-04",
    },
    "Cleaning": {
        "objects": ["spill", "meeting room", "hallway", "waste bin", "window"],
        "actions": ["clean", "sanitize", "remove", "wash", "inspect"],
        "indicator": "IND-05",
    },
    "Security": {
        "objects": ["access badge", "visitor entrance", "parking gate", "camera", "lost item"],
        "actions": ["verify", "monitor", "authorize", "review", "locate"],
        "indicator": "IND-06",
    },
    "Equipment": {
        "objects": ["portable monitor", "office chair", "tool kit", "projector", "power unit"],
        "actions": ["inspect", "replace", "calibrate", "repair", "provide"],
        "indicator": "IND-07",
    },
    "Customer Support": {
        "objects": ["appointment", "information request", "complaint", "service status", "feedback"],
        "actions": ["clarify", "register", "follow up", "review", "respond to"],
        "indicator": "IND-08",
    },
}

LOCATIONS = ["north wing", "main office", "level two", "reception", "storage area"]
QUALIFIERS = ["as soon as possible", "before noon", "when available", "today", "this afternoon"]
TYPES = ["Service Request", "Service Request", "Service Request", "Support Request", "Complaint"]


def noisy(text: str, rng: random.Random) -> str:
    """Add realistic but non-identifying variation to synthetic text."""
    if rng.random() < 0.20:
        text = text.lower()
    if rng.random() < 0.12:
        text = text.replace("please", "pls")
    if rng.random() < 0.10:
        text = text.replace("service", "svc")
    if rng.random() < 0.15:
        text = text.rstrip(".")
    return text


def make_ticket(service: str, template_id: int, rng: random.Random) -> dict[str, str | int]:
    spec = SERVICES[service]
    # Some reports use vocabulary normally associated with another team.
    text_service = rng.choice(list(SERVICES)) if rng.random() < 0.10 else service
    text_spec = SERVICES[text_service]
    ticket_type = rng.choice(TYPES)
    action = rng.choice(text_spec["actions"])
    obj = rng.choice(text_spec["objects"])
    location = rng.choice(LOCATIONS)
    qualifier = rng.choice(QUALIFIERS)

    if ticket_type == "Complaint":
        title = rng.choice([
            f"Complaint: recurring issue with {obj} at {location}.",
            f"The {obj} issue at {location} happened again, please review.",
        ])
        indicator = "IND-09"
    elif ticket_type == "Support Request":
        title = rng.choice([
            f"Please provide support to {action} the {obj} at {location}, {qualifier}.",
            f"Help needed to {action} {obj} in {location} {qualifier}.",
        ])
        indicator = "IND-10"
    else:
        patterns = [
            f"Please {action} the {obj} at {location}, {qualifier}.",
            f"Service needed: {action} {obj} in {location} {qualifier}.",
            f"Request to {action} {obj} - {location} - {qualifier}.",
        ]
        title = rng.choice(patterns)
        indicator = spec["indicator"]

    # Short reports sometimes omit the wording that reveals the request type.
    if rng.random() < 0.14:
        title = f"{action.capitalize()} {obj} at {location}, {qualifier}."

    # A small share represents legitimate cross-category exceptions.
    if rng.random() < 0.035:
        indicator = rng.choice([details["indicator"] for details in SERVICES.values()])

    return {
        "record_id": f"SYN-{template_id:05d}",
        "title": noisy(title, rng),
        "service": service,
        "ticket_type": ticket_type,
        "indicator": indicator,
    }


def main(rows: int = 6000) -> None:
    rng = random.Random(SEED)
    services = list(SERVICES)
    weights = [22, 18, 15, 13, 12, 9, 7, 4]
    records = []
    for idx in range(rows):
        service = rng.choices(services, weights=weights, k=1)[0]
        records.append(make_ticket(service, idx + 1, rng))

    data = pd.DataFrame(records).drop_duplicates(subset=["title", "service", "ticket_type"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data):,} fictional records to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
