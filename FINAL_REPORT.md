# Project report status

This repository does not currently contain a verified real-material benchmark. The checked-in reference CSV uses IDs of the form `SURR-*` and the formula `SurrogateCompound`; these are generated statistical samples, not JARVIS records. Earlier metric tables and optimizer outputs are retained in Git/results as historical artifacts, but they were not reproduced in this audit and must not be presented as real-material findings.

The original evaluation also generated MCS from the complete reference table before splitting, then split the combined real and synthetic table separately. Thus its augmented scores were not measured on the same untouched real test records as its real-only scores. The pipeline has been changed to split first, fit MCS on training rows only, and evaluate both training regimes on the same real test set. That corrected pipeline has **not** been run: the reference cache is rejected by the provenance guard, and required packages are missing in the inspected environment.

The NSGA-II search operates over continuous descriptor vectors without compositions or structures. Such outputs are hypothetical descriptor-space candidates, not identified or verified materials. The included RAG corpus consists of hard-coded sample passages without verifiable source metadata; its self-annotation score is not a scientific RAG quality result.

See [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md), [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md), and [docs/RAG_EVALUATION.md](docs/RAG_EVALUATION.md). Add quantitative findings here only after obtaining verified JARVIS records, installing project dependencies, running the corrected pipeline, and preserving the exact split/configuration used.
