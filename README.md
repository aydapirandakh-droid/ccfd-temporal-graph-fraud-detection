# Temporal Graph-Based Credit Card Fraud Detection

## Project Overview

This project explores graph-based approaches for credit card fraud detection using temporal relationships between financial transactions.

Two implementations are included:

- **Version 1 – Temporal Graph GAT:** An initial graph-based approach using a selected temporal cohort of transactions.
- **Version 2 – Improved Causal Temporal GAT:** An improved approach using the full cleaned dataset, chronological preprocessing, causal temporal connections, and a more carefully designed evaluation strategy.

The project investigates how transaction-level behavioral relationships can support fraud detection beyond traditional tabular classification.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle (ULB Machine Learning Group).

The original dataset contains:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input attributes, including transaction time and amount
- A binary target variable (`Class`)

Duplicate transactions are removed in the improved implementation, resulting in 283,726 transactions.

## Version 1: Temporal Graph GAT

The first implementation constructs a focused temporal cohort around fraudulent transactions.

### Methodology

- Selects fraudulent transactions and nearby normal transactions within a one-hour temporal neighborhood.
- Limits the number of sampled normal transactions around fraud events.
- Builds a directed temporal graph using previous transactions.
- Connects each transaction to up to four behaviorally similar historical transactions within a two-hour window.
- Uses temporal distance as an edge attribute.
- Applies a two-layer GATv2 architecture.
- Uses class-weighted binary cross-entropy for class imbalance.
- Applies chronological 70/15/15 train, validation, and test splits.
- Extracts attention weights and evaluates graph influence through subgraph and edge-ablation experiments.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Selected transactions | 9,676 |
| Fraudulent transactions | 473 |
| Normal transactions | 9,203 |
| Graph nodes | 9,676 |
| Graph edges | 38,694 |

### Test Results

| Metric | Score |
|---|---:|
| PR-AUC | 0.9060 |
| ROC-AUC | 0.9689 |

**Important note:** This implementation uses a selected cohort rather than the complete dataset. Feature scaling is also fitted before the chronological split. Therefore, its reported performance should be interpreted as an exploratory baseline and not directly compared with the improved implementation.

## Version 2: Improved Causal Temporal GAT

The second implementation improves the graph construction and evaluation methodology.

### Methodology

- Removes duplicate transactions and sorts the data chronologically.
- Uses the complete cleaned dataset.
- Applies a chronological 70/15/15 train-validation-test split.
- Fits RobustScaler exclusively on the training period.
- Constructs a causal temporal behavioral graph.
- Connects each transaction to previous transactions within a maximum temporal gap of two hours.
- Uses temporal distance and cosine similarity as edge attributes.
- Verifies that no future-to-past edges exist.
- Implements a lightweight two-layer GATv2 architecture.
- Uses calibrated class weighting based on the training-set imbalance ratio.
- Selects the decision threshold using validation data.
- Evaluates the final model on an unseen test set.

### Graph Statistics

| Metric | Value |
|---|---:|
| Transactions / Nodes | 283,726 |
| Directed edges | 577,041 |
| Node features | 29 |
| Edge features | 2 |
| Future-to-past edge violations | 0 |

### Model Architecture

The model consists of:

- Initial transaction feature projection
- Two GATv2 attention layers
- Multi-head graph attention
- Skip connections
- A fully connected classification head

The model contains 14,145 trainable parameters.

### Test Results

| Metric | Score |
|---|---:|
| PR-AUC | 0.6735 |
| ROC-AUC | 0.8951 |
| Fraud Precision | 86.49% |
| Fraud Recall | 61.54% |
| Fraud F1-score | 71.91% |
| Accuracy | 99.94% |

The classification metrics above use the validation-calibrated threshold of **0.6929**.

### Explainability Analysis

The improved implementation explores graph-based explainability through:

- Attention-weight inspection
- Subgraph-level transaction analysis
- Temporal neighborhood visualization
- Subgraph context ablation
- Attention-based edge deletion experiments

The experiments examine whether historical transaction relationships contribute to model predictions.

**Interpretability limitation:** The measured context-ablation results do not demonstrate a consistent positive confidence drop across evaluated fraud cases. Therefore, attention weights and ablation results are treated as exploratory interpretability analyses rather than conclusive evidence of explanation faithfulness.

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- Pandas
- NumPy
- Scikit-learn
- NetworkX
- Matplotlib
- Seaborn
- KaggleHub
- Google Colab

## Repository Structure

```text
ccfd-temporal-graph-fraud-detection/
│
├── CCFD.Graph.ipynb
├── CCFD_Graph_V2_Improved.ipynb
└── README.md
```

## How to Run

1. Open either notebook in Google Colab.
2. Run the dependency installation cells.
3. Allow KaggleHub to download the Credit Card Fraud Detection dataset.
4. Execute the notebook cells sequentially.
5. Review the model evaluation results and graph explainability outputs.

A CUDA-enabled GPU is recommended for model training.

## Project Notes

This repository documents an experimental progression from an initial temporal graph approach to an improved causal temporal graph model.

The two implementations use different data selection and preprocessing strategies. Their performance metrics should therefore not be interpreted as a direct controlled comparison.

The notebooks are provided for educational and research purposes.
