# Model card

## Intended use
Suggest service, ticket type and performance indicator during review of current operational tickets. The purpose is to reduce classification mistakes that may affect downstream deduction review.

## Not intended for
Calculating or approving deductions, determining responsibility, closing cases autonomously, or treating predictions as final decisions.

## Architecture
Normalized short text uses word and character TF-IDF. Balanced LinearSVC models predict service and ticket type. Indicator models are conditioned on the predicted service; safe fallbacks cover services without sufficient class diversity. Fictional deterministic rules demonstrate hybrid classification.

## Data and privacy
Public records and labels are independently generated and fictional. Private data, taxonomy, identifiers, saved model and organization-specific rules are excluded. Only rounded aggregate historical metrics are reported.

## Risks
Repeated language can inflate random-holdout results; rare classes can be hidden by accuracy; vocabulary can drift; and service errors propagate to the indicator stage.

## Oversight and status
Predictions are suggestions for human review. Monitor reviewer corrections and later-period performance. Previous public artifacts document an older run; the current hierarchical refactor has not yet been executed and will be checked during the portfolio-wide validation.
