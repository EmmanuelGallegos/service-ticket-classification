# Service Ticket Classification

Privacy-safe demonstration of a multi-stage NLP system that classifies short operational requests by service, ticket type, and performance indicator.

## Why this project exists

Operational ticket titles are brief, inconsistent, and highly repetitive. Misclassification can send a case through the wrong review path and create financial exposure when service deductions are assessed. This system supports review of current tickets by suggesting a service, ticket type, and performance indicator; it does not calculate or approve deductions.

The public repository is a clean-room reconstruction. It contains newly written code, fictional categories, and synthetic records only. It does **not** include employer, client, facility, employee, email, ticket identifier, internal route, or original operational data.

## Private-system context

The public repository does not disclose the private dataset size, taxonomy, detailed rules, model artifact, or validation results. Public evaluation uses fictional data only.

## Public demonstration

The synthetic implementation uses:

- text normalization;
- word and character TF-IDF features;
- class-balanced linear support-vector machines;
- a hierarchical prediction flow;
- deterministic business rules for explicit request types;
- service-specific indicator models with safe fallbacks;
- train/test evaluation and confusion-matrix reports.

Fictional service categories include Facilities, IT Support, Logistics, Food Services, Cleaning, Security, Equipment, and Customer Support. Indicator codes are also fictional.

### Previously recorded synthetic test results

After exact-title deduplication, the public dataset contains 5,452 fictional records. On its stratified 20% test partition, the reproducible pipeline obtained:

| Target | Synthetic test accuracy |
| --- | ---: |
| Service category | 91.02% |
| Ticket type — model only | 91.02% |
| Ticket type — final hybrid system | 91.02% |
| Fictional indicator | 85.79% |

These results measure only the deliberately noisy synthetic demonstration. The public code was subsequently refactored to match the hierarchical design, so rerunning the current commit remains pending.

## Project structure

```text
service-ticket-classification/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── professional_metrics.json
├── data/
│   └── synthetic_tickets.csv
├── src/
│   ├── generate_synthetic_data.py
│   ├── modeling.py
│   ├── train_and_evaluate.py
│   └── predict.py
├── models/                 # generated locally, not committed
├── docs/
│   └── model-card.md
└── results/
    ├── synthetic_metrics.json
    ├── service_confusion_matrix.png
    ├── ticket_type_confusion_matrix.png
    └── indicator_confusion_matrix.png
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/train_and_evaluate.py
python src/predict.py data/current_tickets.csv --output results/predictions.csv
```

Inference input requires a `title` column. Authorized current-ticket files must remain local.

## Interpreting the results

Stored synthetic results document a previous public run; they do not validate the refactored commit and are not substitutes for production monitoring. See the [model card](docs/model-card.md). The generated dataset intentionally includes spelling variation, short descriptions, overlapping vocabulary, and imbalanced categories.

## Privacy statement

All public examples and labels are fictional. Aggregate professional metrics are rounded, and the original organization's identity, taxonomy, records, and operating rules are excluded.

## Author

Emmanuel Gallegos Montiel — Actuary and Applied Artificial Intelligence graduate student.
