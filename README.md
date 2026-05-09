# 🧪 Synthetic Data Generation Pipeline (Distilabel + GraphGen)

> Production-grade synthetic instruction data pipeline for domain-adapted SFT, using Distilabel for text generation and automated quality scoring.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Distilabel](https://img.shields.io/badge/Framework-Distilabel-orange.svg)](https://github.com/argilla-io/distilabel)

```mermaid
flowchart LR
    S[Seed Topics] --> E[Evol-Instruct]
    E --> G[Generate Responses]
    G --> QF[Quality Filter]
    QF --> |PPL < τ| DS[Dedup + Diversity]
    DS --> |MinHash + Embedding| V[Validation]
    V --> |Business Logic| OUT[train.jsonl]
```

## 🎯 Problem

High-quality instruction data is the bottleneck for domain SFT. Manual curation is expensive ($5-15 per example via human annotation). Synthetic generation at scale requires principled quality control to avoid "model collapse" — where low-quality synthetic data degrades model performance.

## 🧮 Mathematical Foundation

### Self-Instruct Quality Score

$$Q(x,y) = \alpha \cdot \text{PPL}^{-1}(y|x) + \beta \cdot \text{Div}(x) + \gamma \cdot \text{BL}(x,y)$$

where PPL = perplexity (fluency), Div = diversity score, BL = business logic pass rate.

### Evol-Instruct Complexity Escalation

Given seed instruction $x_0$, generate increasingly complex variants:

$$x_{k+1} = \text{LLM}(\text{evolve}(x_k, \text{strategy}_k))$$

Strategies: deepening, concretizing, adding constraints, multi-step reasoning.

### Deduplication: MinHash LSH

$$\text{Sim}(A, B) = \frac{|A \cap B|}{|A \cup B|} \approx P[\min(h(A)) = \min(h(B))]$$

Approximate Jaccard similarity via $k$ hash functions with $b$ bands of $r$ rows. Threshold: $t = (1/b)^{1/r}$.

### Embedding Diversity (Determinantal Point Process)

$$P(\mathcal{S}) \propto \det(L_\mathcal{S}), \quad L_{ij} = K(\mathbf{e}_i, \mathbf{e}_j)$$

DPP sampling maximizes diversity in embedding space — ensures training data covers the full domain.

### IFD Score (Instruction-Following Difficulty)

$$\text{IFD}(x, y) = \frac{\log p_\theta(y \mid x)}{\log p_\theta(y)}$$

High IFD → instruction significantly helps predict response (high-quality pair).
Low IFD → response is generic regardless of instruction (discard).

### Perplexity-Based Filtering

$$\text{PPL}(y \mid x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(y_t \mid y_{<t}, x)\right) < \tau$$

### Data Scaling Law

$$\text{Performance} \propto N_{\text{eff}}^{\alpha}, \quad N_{\text{eff}} = N \cdot (1 - r_{\text{dup}}) \cdot q_{\text{avg}}$$

where $r_{\text{dup}}$ = deduplication rate, $q_{\text{avg}}$ = average quality score.

## 🏥 Enterprise Pharma Application

In enterprise pharma, I generate synthetic promotional scenarios for model training:

| Production Need | This Pipeline |
|---|---|
| Training data for MMM explanation model | Evol-Instruct from seed topics |
| Privacy-safe HCP targeting scenarios | Synthetic persona + engagement data |
| Diverse budget optimization examples | Parameterized scenario generation |
| Business logic validation | Automated checks (ROI ranges, CI format) |

**Key insight:** In enterprise pharma, I observed that 500 human-validated Q&A pairs outperform 5,000 unfiltered synthetic ones. This pipeline operationalizes that finding with automated quality scoring.

## 🚀 Quickstart

```bash
git clone https://github.com/fab-admasu/synth-data-distilabel-pipeline.git
cd synth-data-distilabel-pipeline
pip install -r requirements.txt

# Generate seed topics
python scripts/generate_seeds.py --domain pharma_analytics --n_seeds 50

# Run Evol-Instruct pipeline
python scripts/evolve_instructions.py --seeds data/seeds.jsonl --n_evolutions 3

# Quality filter + dedup
python scripts/quality_filter.py --data data/evolved.jsonl --ppl_threshold 50

# Validate
python scripts/validate_quality.py --data data/filtered.jsonl
```

## 📊 Evaluation

| Stage | Count | Avg Quality | Diversity | Business Logic Pass |
|---|---|---|---|---|
| Raw seeds | 50 | — | — | — |
| After Evol-Instruct (3 rounds) | 500 | 0.62 | 0.45 | 71% |
| After PPL filter (τ=50) | 380 | 0.78 | 0.52 | 89% |
| After dedup (MinHash t=0.7) | 340 | 0.79 | 0.68 | 89% |
| After business logic filter | **310** | **0.85** | **0.71** | **100%** |

### SFT Impact (downstream)

| Training Data | Instruction Following | JSON Valid | Domain Accuracy |
|---|---|---|---|
| 1000 unfiltered synthetic | 72% | 81% | 69% |
| 310 quality-filtered (this pipeline) | **86%** | **95%** | **84%** |

## License

MIT
