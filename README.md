# Federated Illusion: Privacy Leakage in Federated Graph Unlearning via Multi-Level Geometric Audit

## Core Thesis

> **Federated aggregation creates a stronger privacy illusion** — FedAvg dilutes confidence-level signals (Conf AUC ≈ 0.5), but L2 geometric audit reveals persistent leakage at global, local, and cross-client levels. Only complete federated retraining provides true unlearning guarantees.

| Audit Level | Approximate Methods | FedRetrain (Gold) |
|-------------|--------------------:|------------------:|
| **Global Conf AUC** | ~0.48 | ~0.50 |
| **Global L2 AUC** | **>0.70** | ~0.51 |
| **Local L2 AUC** | **>0.85** | ~0.51 |
| **Cross-Client L2 AUC** | **>0.60** | ~0.51 |

---

## Research Questions

| RQ | Question | Expected Finding |
|----|----------|-----------------|
| **RQ1** | Does FedAvg create a stronger confidence illusion? | Gap (L2-Conf) > 0.20, stronger than centralized |
| **RQ2** | Does unlearning signal propagate across client boundaries? | Correlated with cross-client edge count (r > 0.5) |
| **RQ3** | Which audit level leaks most? | Local > Global > Cross-Client |
| **RQ4** | Can a limited auditor detect remote unlearning? | Requires >= 5 cross-edges |
| **RQ5** | Client-level vs node-level unlearning? | Client-level better but still leaky |

---

## Quick Start

```bash
# Install dependencies
pip install torch torch-geometric scikit-learn numpy pandas scipy matplotlib networkx

# Verify environment (centralized, ~5 min)
python experiments/run_pilot.py

# Run federated pilot (after implementation, ~5 min)
python experiments/federated/run_fed_pilot.py

# Full federated experiment (~8-12h with GPU)
python experiments/federated/run_fed_main.py --trials 50
```

---

## Project Structure

```
Fed_MIA/
├── src/
│   ├── attacks/          # Hub-Ripple MIA attack + evaluation metrics
│   ├── models/           # GCN, GAT, GraphSAGE architectures
│   ├── unlearning/       # 6 centralized unlearning methods (foundation)
│   ├── utils/            # Data loading, graph operations, common tools
│   └── federated/        # [TODO] Federated client-server + multi-level attack
├── configs/              # Experiment configuration
├── experiments/          # Centralized experiment scripts (completed)
├── experiments/federated/# [TODO] Federated experiment scripts
├── paper/                # Centralized paper LaTeX (reference)
├── docs/                 # Experiment design documents
├── tables/               # Centralized experiment results (CSV)
├── results/              # Centralized results (figures, logs)
└── data/                 # Dataset cache (Cora, CiteSeer, PubMed, etc.)
```

---

## Experiment Matrix

### Federated (Primary)

| Dimension | Values | Count |
|-----------|--------|------:|
| Datasets | Cora, CiteSeer, PubMed, Chameleon, Squirrel | 5 |
| GNN | GCN-2L | 1 |
| Partition | Metis, Random | 2 |
| Clients | 3, 5, 10 | 3 |
| Distribution | IID, Label-Skew-0.5 | 2 |
| Unlearning | FedRetrain, FedGNNDelete, FedGraphEraser | 3 |
| Granularity | Node-level, Client-level | 2 |
| Trials | 50 | - |

**Total: 18,000 trials** + 2,950 extended (threat model + edge ratio + re-agg ablation)

### Centralized (Completed Baseline)

5 datasets x 4 GNN models x 6 methods x 100 trials = **12,000 trials**

---

## License

MIT License
