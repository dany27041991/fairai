# Scientific validity notes (IEEE Software draft)

Observations to resolve before submission. Evidence from `fairai` artifacts and the current `.tex`.

## Setup (facts)

| Fact | Value |
|------|--------|
| Training pool | ~50 409 leakage-safe sentences (StereoSet **test** strings removed) |
| Train / val | **45 368 / 5 041**, stratified 90/10, seed 42 (`van_split_seed42_*.jsonl`) |
| Held-out preference eval | StereoSet (inter + intra), separate from classification val |
| Released checkpoint | Optuna VAN v2; product numbers below |
| Matched suite | Same fixed split + same iCAT protocol; 3 seeds (`rigorous_*`) |

---

## 1. Baseline comparison (same split / protocol / preprocessing)

**Problem.** An earlier draft mixed (a) a historical “StereoSet test F1 ≈ 0.956” with (b) baselines’ validation macro-F1 from matched retrains. That is not a fair head-to-head.

**Rule.** Never put two models in the same F1 column unless the F1 is the same dataset, split, and metric.

**Treatment (two tables, both honest):**

| Table | What it answers | Protocol |
|-------|-----------------|----------|
| **A – Deployment (released)** | What the published weights do | **leak-free** select-options F1 (neutral label 2; 0 exact overlap vs train/val + select train) + official StereoSet **test** adapted iCAT |
| **B – Matched ablations** | What architecture / loss add | Fixed train/val JSONL; F1 = **val macro-F1**; avg iCAT on StereoSet **`subset=test`** |

**Table B SoT (VAN Optuna reference vs equal-budget baselines):** `optuna_product_compare_agg.json`  
VAN = **released Optuna checkpoint** (Table A weights). CE / −BPL = equal-budget Optuna (100 trials, WHM train-iCAT) 3-seed finals.  
**Same eval plane as Table A:** leak-free F1 + StereoSet-**test** iCAT.

| Family | F1 leak-free | iCAT inter | iCAT intra |
|--------|-------------:|-----------:|-----------:|
| **VAN Optuna released** | **0.962** | **96.28** | **96.71** |
| CE Optuna | 0.746±0.005 | 88.7±7.9 | 94.5±4.4 |
| −BPL Optuna | 0.666±0.069 | 79.3±18.8 | 91.4±6.0 |
| Frozen (no Optuna) | 0.541 | 81.32 | 78.53 |

**Result:** Released Optuna VAN leads on the shared product plane.  
Archive (from-scratch VAN re-search): `optuna_equal_product_compare_agg.json` — not Table B SoT.

**Paper claim:** Table A/B share metrics; Table B compares the Optuna-selected VAN to baselines given the same search budget.

---

## 2. Adapted iCAT validity

**Original StereoSet** (Nadeem et al.): LM ranks candidates by **language-model likelihood** within a triplet.

**Our adaptation (3-class classifier):** each candidate is scored **independently**; no joint LM over the triplet.

\[
\begin{aligned}
\mathrm{LM} &= 100\cdot\frac{1}{N}\sum_i \mathbf{1}\!\bigl[\max(p^{(i)}_{\mathrm{stereo}},p^{(i)}_{\mathrm{anti}}) > p^{(i)}_{\mathrm{unrel}}\bigr] \\
\mathrm{SS} &= 100\cdot\frac{1}{N}\sum_i \mathbf{1}\!\bigl[p^{(i)}_{\mathrm{stereo}} > p^{(i)}_{\mathrm{anti}}\bigr] \\
\mathrm{iCAT} &= \mathrm{LM}\cdot\frac{\min(\mathrm{SS},\,100-\mathrm{SS})}{50}
\end{aligned}
\]

Aggregation used everywhere in code/docs:

\[
\mathrm{avg\_icat} = \tfrac{1}{2}\bigl(\mathrm{iCAT}_{\mathrm{inter}}+\mathrm{iCAT}_{\mathrm{intra}}\bigr)
\]

Concatenation: inter = `context [SEP] sentence`; intra = sentence only (`EVAL_ICAT.md`).

**Controlled checks (required in paper or supplement):**

| Case | Expected |
|------|----------|
| Always \(p_s > p_a > p_u\) | LM→100, SS→100, iCAT→0 |
| Always \(p_a > p_s > p_u\) | LM→100, SS→0, iCAT→0 |
| Always \(p_s \approx p_a > p_u\) with SS≈50 | LM high, iCAT≈LM |
| Always \(p_u\) largest | LM→0, iCAT→0 |
| Unit tests | `tests/unit/test_icat_formula.py` |

State explicitly: adapted iCAT is a **preference fairness probe for a classifier**, not a drop-in replica of the original LM iCAT.

---

## 3. iCAT value inconsistency (98.13 / 97.05 / 96.1)

**Root cause:** three different numbers from **two different runs/aggregations** were printed as if they were one evaluation.

| Number | Provenance | Meaning |
|-------:|------------|---------|
| **98.13** | Thesis notebook (historical print) | \(\mathrm{iCAT}_{\mathrm{inter}}\) from LM=98.18, SS=49.98 |
| **97.05** | Same historical print | \(\mathrm{iCAT}_{\mathrm{intra}}\) from LM=97.23, SS=49.91 |
| **96.1** | Reproducible `fairai icat` on released v2, **subset=all** | \(\mathrm{avg\_icat}=(96.0505+96.1441)/2 = 96.097\) |

If one used the thesis pair, the consistent average would be \((98.13+97.05)/2 = 97.59\), **not** 96.1. So 96.1 cannot be described as the average of 98.13 and 97.05.

**Reproducible SoT (released VAN v2; applied in `ieee_software_stereotypes.tex`):**

| Subset | inter iCAT | intra iCAT | avg_icat | Source |
|--------|-----------:|-----------:|---------:|--------|
| **all** (n=12 687) | 96.05 | 96.14 | **96.10** | `baseline_van_v2_icat.json` |
| **test** (n=3 816) | 96.28 | 96.71 | **96.49** | `icat_test_verify.json` |

Per-bias test (same file):

| Bias | iCAT inter | iCAT intra |
|------|-----------:|-----------:|
| gender | 86.47 | 85.64 |
| race | 96.95 | 91.66 |
| profession | 96.15 | 96.59 |
| religion | **75.62** | 95.87 |

\(\Delta_{\mathrm{iCAT}}\) inter \(\approx 21.3\); pooled test gap \(\approx 10.3\).

**Done in tex:** dropped 98.13/97.05; All row = test 96.28/96.71; prose states 96.10 is subset=all only.

**Table A F1 by bias (leak-free SoT):**

| Protocol | Resource | Result |
|----------|----------|--------|
| iCAT | Official StereoSet **test** | avg_icat **96.49** (inter 96.28 / intra 96.71) |
| F1 | **Leak-free** select-options (`f1_test_leakfree.jsonl`; unrelated→neutral label 2); `context [SEP] option` vs `label` | macro-F1 **0.962**; gender/race/profession/religion **0.974 / 0.966 / 0.952 / 1.000** |
| Not used (Table A) | Official StereoSet `gold_label` | F1 ≈ 0.41 (wrong class-2 semantics) |
| Legacy only | Full select_options test (`select_options_test_f1.json`) | F1 ≈ 0.964 with ~48% exact overlap vs train — **not** Table A |

Primary artifact: `artifacts/results/metrics/f1_test_leakfree_released.json`  
Leakage report: `artifacts/analysis/f1_test_leakfree_report.json` (`overlap_exact_vs_forbidden: 0`).

**Optuna 0.92 / 98.2:** historical trial selection readout only — not Table A/B SoT (stated in tex).

**PPO P6:** thesis controlled MC experiment; fairai ships smoke/env/reward only (GAP-F5; stated in Limitations).

---

## 4. “Variance-Aware” claim

**Problem.** The name suggests the model **minimizes inter-group performance disparity**. Training actually optimizes a weighted harmonic mean of **global** F1 and **global** avg_icat (weight 1.88 on F1). That is joint accuracy–fairness selection, **not** an explicit inter-group variance penalty.

**Evidence against a strong claim:** large gaps remain across bias types (see §3).

**Treatment (recommended):**

1. **Ridimensionare** the claim in title/abstract/Tool-2 naming prose:
   - Keep acronym VAN if desired, but define it as selected on **both detection quality and StereoSet preference fairness (iCAT)**.
   - Do **not** claim that Optuna “keeps inter-group disparity low” unless that quantity is the objective.
2. **Optional measurement** (if the name is kept): report
   \[
   \Delta_{\mathrm{iCAT}} = \max_g \mathrm{iCAT}_g - \min_g \mathrm{iCAT}_g
   \]
   (and/or \(\Delta_{\mathrm{F1}}\)) on StereoSet test, and treat it as a **diagnostic**, not as proof that training minimized it.
3. Secondary “variance” (MC dropout abstention) may stay as an **operational** note, clearly separated from the name’s justification.

Suggested abstract phrasing: *“…a compact classifier selected jointly on accuracy and a StereoSet fairness score (iCAT)…”* — avoid “variance-aware” in the abstract unless \(\Delta\) is defined and reported in the body.

---

## Checklist before submission

### Scientific (closed in current draft)

- [x] Table A = released only; Table B = matched only; captions name the metric/split
- [x] Table A F1 = **leak-free** select-options (0 exact overlap); not the 48%-overlap legacy set
- [x] Metrics § no longer claims 322/966 as Table A F1 resource
- [x] Leak-free construction discloses 136 test + 14 val-promoted prompts
- [x] Table A: argmax only (no gates/MC); religion n=9 caveat in caption
- [x] Table B iCAT reported on StereoSet **`subset=test`** (re-eval); historical `all` archived
- [x] Table B caption + numbers from clean retrain (`icat_subset: train`); Frozen LR no early-stop; peeked historical archived
- [x] Dead YAML keys `lr_multiplier` / `ema_decay` marked unused in `van_v2.yaml` + `MODEL_VAN.md`
- [x] GAP-D4: 17 residual train overlaps disclosed; strict export noted
- [x] Single iCAT SoT (fairai); formula + avg definition + concat rule in text
- [x] Controlled iCAT examples in Metrics section (+ unit tests in repo)
- [x] 98.13 / 97.05 removed; test avg_icat=96.49; all=96.10 labeled separately
- [x] Optuna 0.92 / 98.2 labeled as selection-time diagnostics, not Table A/B
- [x] Split counts 45 368 / 5 041 (seed 42) in tex + this note
- [x] “Variance-Aware” softened; \(\Delta_{\mathrm{iCAT}}\) reported
- [x] Per-bias iCAT from `icat_test_verify.json`; Table A F1 from `f1_test_leakfree_released.json`
- [x] GitHub URL → `https://github.com/dany27041991/fairai` (v0.1.0)
- [x] PPO P6 rates de-emphasized as thesis-historical (not open multi-seed CSV)
- [x] MC abstention claim aligned to shipped labeling (11/6369), not unverified 2/1911
- [x] Class weights in `train_van` use **train fold only** (code fix)
- [x] `suite_unified.yaml` eval dataset = `f1_test_leakfree.jsonl` (SoT 450)

### Still open (protocol / optional retrain — not editorial)

- [x] **Retrain** `suite_rigorous_clean` (`rigc-*`) archived as fixed-HP interim.
- [x] **Table B** = Optuna released VAN vs equal-budget Optuna CE/−BPL on leak-free F1 + test iCAT (`optuna_product_compare_agg.json`: VAN 0.962 / 96.28 / 96.71 leads).
- [ ] Optional: implement dual-LR / EMA, or delete unused keys from YAML entirely
- [ ] Source-attribution JSON for the 39 300 per-source counts (or soften numbers)
- [ ] Human neutral audit beyond heuristic junk rate

### Editorial (still open — human)

- [ ] IEEE DataPort DOI (`\tbd` + `\nota` in tex)
- [ ] Sellalab program name / funding `\nota`
- [ ] Author biographies `\nota` (Vergallo / Lotti / Mainetti)
