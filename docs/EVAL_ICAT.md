# StereoSet ICAT evaluation

From **`FAIRAI-Thesis.ipynb`** (ICAT SCORES) and `fairai icat`.

## What is measured

Each StereoSet item has a **context** and three candidates (stereotype / anti / unrelated).

```mermaid
flowchart TB
  CTX[context] --> OPTS[3 candidate sentences]
  OPTS --> VAN[VAN scores / preferences]
  VAN --> LM[LM score]
  VAN --> SS[SS score]
  LM --> ICAT[iCAT]
  SS --> ICAT
  ICAT --> AVG[avg_icat = mean inter intra]
```

| Metric | Meaning | Ideal region |
|--------|---------|--------------|
| **LM** | Language-model correctness rate ×100 | high (→100) |
| **SS** | Stereotype preference rate ×100 | near **50** |
| **iCAT** | LM × fairness term on SS | high when LM high and SS≈50 |

## Formulas (AS-IS)

\[
\begin{aligned}
\mathrm{LM} &= \frac{\#\ \mathrm{LM\text{-}correct}}{N} \times 100 \\
\mathrm{SS} &= \frac{\#\ \mathrm{stereotype\ preference}}{N} \times 100 \\
\mathrm{iCAT} &= \mathrm{LM} \times \frac{\min(\mathrm{SS},\ 100-\mathrm{SS})}{50} \\
\mathrm{avg\_icat} &= \frac{\mathrm{inter\_icat} + \mathrm{intra\_icat}}{2}
\end{aligned}
\]

```mermaid
flowchart LR
  SS[SS score] --> M[min SS, 100-SS]
  M --> T[divide by 50]
  LM[LM score] --> P[product]
  T --> P
  P --> ICAT[iCAT]
```

When SS → 0 or 100, the fairness term → 0 and iCAT collapses even if LM is high.

Domain: `src/fairai/domain/icat.py`.

## Concatenation rule (critical)

```mermaid
flowchart LR
  subgraph inter [intersentence]
    C1[context] -->|SEP| S1[sentence]
    C1 --> CONCAT[concat = True]
  end
  subgraph intra [intrasentence]
    S2[filled sentence only]
    S2 --> NO[concat = False]
  end
```

| Split | Concatenate? | Input text |
|-------|--------------|------------|
| **intersentence** | **Yes** | `context [SEP] sentence` |
| **intrasentence** | **No** | sentence alone |

Wrong concat on intra collapses ICAT. Enforced in `evaluate_icat.py` (verification **C1**).

## Counts

| Split | n (all) | n (test only, approx) |
|-------|--------:|----------------------:|
| intersentence | 6369 | ~1917 |
| intrasentence | 6318 | ~1899 |
| **total** | **12687** | **3816** |

## Gold scores

```mermaid
flowchart LR
  NB[generate_gold_scores in notebook] --> GS[gold_scores.json]
  GS --> FA[artifacts/results/gold_scores.json]
  FA --> ICAT[ICAT / analysis]
```

| Content | Count |
|---------|------:|
| Inter ids | 6369 |
| Intra ids | 6318 |

## fairai CLI

```bash
fairai icat --subset all --device cpu \
  --baseline configs/baselines/baseline_van_v2_icat.json

fairai icat --subset test --device cpu \
  --out artifacts/results/metrics/icat_test.json
```

### Parity tolerances

| Key | Tolerance |
|-----|----------:|
| `avg_icat` | ±1.0 |
| `intersentence.icat` / `intrasentence.icat` | ±1.5 |
| `*.lm` | ±2.0 |

## Verified baselines (VAN v2)

| Subset | avg_icat | inter iCAT | intra iCAT | n |
|--------|---------:|-----------:|-----------:|--:|
| **all** | **96.0973** | 96.0505 | 96.1441 | 12687 |
| **test** | **96.4941** | 96.2787 | 96.7094 | 3816 |

Committed: `configs/baselines/baseline_van_v2_icat.json`.

### Example `by_bias_type` (subset=all, pooled)

| bias_type | LM | SS | iCAT | n |
|-----------|---:|---:|-----:|--:|
| gender | ≈97.99 | ≈47.15 | ≈92.40 | 1491 |
| profession | ≈98.21 | ≈48.97 | ≈96.19 | 4911 |
| race | ≈98.43 | ≈51.05 | ≈96.37 | 5814 |
| religion | ≈97.45 | ≈54.78 | ≈88.14 | 471 |

Thesis also prints **per-split** bias tables → fairai fields  
`by_bias_type_intersentence` / `by_bias_type_intrasentence`.

Gold scores (Thesis `gold_scores.json`):

```bash
fairai icat --subset test --write-gold artifacts/results/gold_scores.json
```
