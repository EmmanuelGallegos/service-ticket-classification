# Service Ticket Classification

Privacy-safe demonstration of a multi-stage NLP system that classifies short operational requests by service, ticket type, and performance indicator.

## Why this project exists

Operational ticket titles are brief, inconsistent, and highly repetitive. Categories with few examples are especially difficult to learn. This project demonstrates an iterative workflow in which historical text is collected, cleaned, reviewed, and used to improve classification coverage over time.

The public repository is a clean-room reconstruction. It contains newly written code, fictional categories, and synthetic records only. It does **not** include employer, client, facility, employee, email, ticket identifier, internal route, or original operational data.

## Professional-system results

The original private system was developed using more than 400,000 historical records. Its final recorded 80/20 historical validation produced the following aggregate results:

| Target | Validation accuracy |
| --- | ---: |
| Service category | ~99.0% |
| Ticket type — model only | ~99.8% |
| Ticket type — final hybrid system | ~99.8% |
| Performance indicator — final system | ~98.1% |

These figures describe historical validation, not independently verified production performance. Repeated or near-duplicate titles may make a random holdout easier than a temporal or grouped evaluation. No per-service results or confidential labels are disclosed.

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

### Synthetic test results

After exact-title deduplication, the public dataset contains 5,452 fictional records. On its stratified 20% test partition, the reproducible pipeline obtained:

| Target | Synthetic test accuracy |
| --- | ---: |
| Service category | 91.02% |
| Ticket type — model only | 91.02% |
| Ticket type — final hybrid system | 91.02% |
| Fictional indicator | 85.79% |

These results measure only the deliberately noisy synthetic demonstration.

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
│   └── train_and_evaluate.py
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
```

## Interpreting the results

Synthetic results prove that the public workflow runs end to end; they are not substitutes for the private system's historical metrics. The generated dataset intentionally includes spelling variation, short descriptions, overlapping vocabulary, and imbalanced categories.

## Privacy statement

All public examples and labels are fictional. Aggregate professional metrics are rounded, and the original organization's identity, taxonomy, records, and operating rules are excluded.

## Author

Emmanuel Gallegos Montiel — Actuary and Applied Artificial Intelligence graduate student.
