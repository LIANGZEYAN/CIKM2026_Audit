# Paper title

Transforming User Actions into Audit Signals: Participatory Auditing to Improve Search Results

## Installation

```bash
pip install numpy pandas torch transformers scipy matplotlib seaborn python-terrier ir-measures
```

## Complete Pipeline

### Step 1: Generate SBR Rankings
```bash
python ipssim_version_1.py ColBERT_ranking.csv 5 1.0 --output SBR_ranking.csv
```
Applies semantic similarity to rerank ColBERT results.

**Formula:** `Score_SBR = Score_ColBERT × (1 + α × AvgSim)`

---

### Step 2: Strategic Document Selection (User Study 1)
```bash
python strategy_merge_version1.py ColBERT_rankings.csv SBR_rankings.csv --top_k 4 --output ColBERT_SBR_merge_set.csv
```
Merges **ColBERT** and **SBR** rankings to select documents for User Study 1.

**Selects:** Top-4 ColBERT + Top-4 SBR + 1 easy negative (9 docs per query)

---

### Step 3: Human Evaluation (User Study 1)
```bash
cd evaluation_interface
cp ../evaluation_set_v1.csv strategic_selection_results_version1.csv
python app.py
```
Web interface comparing **ColBERT vs SBR**. Open browser at `http://localhost:5000`

**Output:** User click logs(User_Study_1.sql)

---

### Step 4: Interaction Data Analysis
```bash
python click_data_analysis.py
```
Analyzes click logs from User Study 1.

**Output:** User statistics + visualizations

---

### Step 5: Optimize Alpha & Generate IPSsimRF Rankings
```bash
python ipssimrf_optimization.py
```
Optimizes α using audit signals (real user clicks) and generates IPSsimRF rankings.

**Edit paths in script:**
```python
colbert_rankings_path = "ColBERT_rankings.csv"
real_clicks_path = "User_Study_1.sql"        # From Step 3
```

**Output:** 
- Optimal α value (e.g., 2.6429)
- `IPSsimRF_ranking.csv` (IPSsimRF rankings)

**Formula:** `Score_IPSsimRF = Score_ColBERT × (1/ρd + α × AvgSim)`

---

### Step 6: Strategic Document Selection (User Study 2)
```bash
python strategy_merge_version2.py SBR_rankings.csv IPSsimRF_ranking.csv --top_k 4 --output SBR_IPSsimRF_merge_set.csv
```
Merges **SBR** and **IPSsimRF** rankings to select documents for User Study 2.

**Selects:** Top-4 SBR + Top-4 IPSsimRF + 1 easy negative (9 docs per query)

---

### Step 7: Human Evaluation (User Study 2)
```bash
cd evaluation_interface
cp ../SBR_IPSsimRF_merge_set.csv
python app.py
```
Web interface comparing **SBR vs IPSsimRF**. Open browser at `http://localhost:5000`

**Output:** User click logs(User_Study_2.sql)

---

### Step 8: Interaction Data Analysis (User Study 2)
```bash
python click_data_analysis.py
```
Analyzes click logs from User Study 2 (same script as Step 4).

**Edit paths in script:**
```python
click_data_path = "User_Study_2.sql"      # From Step 7
label_data_path = "SBR_IPSsimRF_merge_set.csv"      # From Step 6
```

**Output:** User statistics + visualizations comparing SBR vs IPSsimRF

---

### Step 9: Dwell Time Analysis (RQ 5.3)
```bash
python dwell_time_analysis.py click_logs.sql --output dwell_time_summary.csv
```
Analyzes user engagement duration for PASSAGE_SELECTION and OPEN_DOC events.

**Answers:** How long do users engage with documents?

**Output:**
- Mean and median dwell times (seconds)
- Distribution statistics (std, min, max, quartiles)
- Optional CSV summary

**Can be applied to either User Study 1 or 2 data.**

---

## Reproduce nDCG Results

We provide minimal files for direct evaluation without running the full pipeline.

### Files Included
- `ipssimrf_ranking.csv` - IPSsimRF rankings (qid, docno, score)
- `evaluate_ndcg.py` - Evaluation script

### Quick Evaluation
```bash
pip install python-terrier ir-measures
python evaluate_ndcg.py ipssimrf_ranking.csv
```

### Expected Output
```
nDCG@1   : X.XXXX
nDCG@3   : X.XXXX
nDCG@5   : X.XXXX
nDCG@10  : X.XXXX
nDCG@30  : X.XXXX
nDCG@50  : X.XXXX
nDCG@100 : X.XXXX
```

---

## Key Formulas

**SBR (Step 1):**
```
Score_SBR(d) = Score_ColBERT(d) × (1 + α × AvgSim(d, D_top))
```

**IPSsimRF (Step 5):**
```
Score_IPSsimRF(d) = Score_ColBERT(d) × (1/ρd + α × AvgSim(d, D_top))
```

Where:
- `ρd`: Position-based propensity score (removes position bias)
- `α`: Optimized weight parameter (from Step 5)
- `AvgSim`: Average semantic similarity to top-k documents

---

## Workflow Summary

```
Step 1: Generate SBR rankings
   ↓
Step 2: Merge ColBERT + SBR → Evaluation Set v1
   ↓
Step 3: User Study 1 (ColBERT vs SBR)
   ↓
Step 4: Analyze clicks from User Study 1
   ↓
Step 5: Optimize α + Generate IPSsimRF rankings
   ↓
Step 6: Merge SBR + IPSsimRF → Evaluation Set v2
   ↓
Step 7: User Study 2 (SBR vs IPSsimRF)
   ↓
Step 8: Analyze clicks from User Study 2
   ↓
Step 9: Dwell time analysis (RQ 5.3)
   ↓
Reviewers: Evaluate nDCG directly
```

**Two User Studies:**
- **User Study 1** (Steps 2-4): Compare **ColBERT** vs **SBR**
- **User Study 2** (Steps 6-8): Compare **SBR** vs **IPSsimRF**

**Additional Analysis:**
- **Step 9**: Can be applied to either study to analyze user engagement duration

---

## Notes

- **Steps 4 & 8** use the same script (`click_data_analysis.py`) for interaction analysis
- **Step 9** (`dwell_time_analysis.py`) can analyze either User Study 1 or 2 data
- Just update input file paths to analyze different datasets
- **Step 5** produces both optimal α and complete IPSsimRF rankings
- For detailed parameters, see individual script docstrings

---

## Naming: paper "RD" ↔ code "rd"

The paper (Section 3.3, Eq. 3) uses **Rank Deficit (RD)** as the per-document
audit-signal statistic that drives α-tuning in Step 5. In `Utils/IPSsimRF_optim.py`
the corresponding columns and arguments are `rd`, `rd_threshold`, `is_high_rd`,
etc. (the file also exposes `analyze_rd_distribution(...)`).

---

## Resubmission experiments (`experiments/`)

The `experiments/` directory contains the additional analyses cited in
Sections 5 and 5.1 of the paper. Each script writes its outputs (CSV / TeX /
log) alongside itself, so the included artefacts are the exact numbers
referenced in the paper. Inputs are hardcoded to the original development
paths under `/mnt/primary/...`; to re-run, edit the path constants at the top
of each script.

| Script | Section / Table cell | What it produces |
| --- | --- | --- |
| `a1_significance.py` + `a1_format_table.py` | Table 1 (significance markers) | Paired bootstrap (B = 10,000, shift method) + Bonferroni over 21 tests, formats Table 1 LaTeX |
| `a23_5fold_cv.py` | Table 1 row "IPSsimRF (5-fold CV)" | Per-fold α tuning + held-out nDCG (Section 5.1, robustness check (i)) |
| `a4_partial_sbr_vs_random.py` | Section 3.1, end | SBR-vs-random low-rank sampling comparison (1.50 vs 0.39 mean label, 0.83 vs 0.74 SimCSE) |
| `a4b_random_auditor.py` | Table 1 row "IPSsimRF (random D_top^user)" | 5-seed random-D_top^user ablation (Section 5.1, robustness check (iii)) |
| `a8_clicks_only.py` | Section 5, RQ3 paragraph | Stage-1 OPEN_DOC clicks-only variant of IPSsimRF |
| `b6_absolute_rates.py` | Section 5 click/selection rates | 60.7 / 68.2 % click rates; 54.6 / 60.7 % selection rates per study |
| `c3_asr_max_features.py` | Section 5, ASR-Max characterisation | 13 ASR-Max docs at 92.9th ColBERT percentile, SimCSE 0.82 vs 0.84 |

The ColBERT-top-D_top^user ablation (Table 1 row "IPSsimRF (ColBERT-top D_top^user)")
is produced by `a23_5fold_cv.py`'s `optimize_alpha_ndcg` path (the "A3" variant
in that script — see its module docstring), so no separate file is needed.

---

## Citation

[Citation information will be added after publication]
