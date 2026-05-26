"""
A1 — Statistical significance tests on nDCG (CIKM 2026 resubmission).

Computes per-query nDCG@{1,3,5,10,30,50,100} for ColBERT, SBR, IPSsimRF
on TREC DL 2021 (53 queries), then runs paired bootstrap (B=10000),
paired t-test, and Cohen's d for the three pairwise comparisons.
Applies Bonferroni correction across 21 tests (3 pairs × 7 cutoffs).

Inputs:
- /mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv
- /mnt/primary/Trec-llm/utils/case_study_rankings/sbr_for_evaluation.csv
- /mnt/primary/Trec-llm/utils/case_study_rankings/ipssimrf_for_evaluation.csv
- /mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels

Outputs:
- a1_per_query_ndcg.csv         (qid, system, cutoff, ndcg)
- a1_significance_results.csv   (cutoff, comparison, mean_a, mean_b, delta, t_pvalue, bootstrap_pvalue, cohens_d, bonferroni_significant)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import pyterrier as pt
if not pt.started():
    pt.init()
from ir_measures import nDCG, iter_calc

# ----- Config -----
RANKINGS_DIR = "/mnt/primary/Trec-llm/utils/case_study_rankings"
QRELS_PATH = "/mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels"
OUT_DIR = "/mnt/primary/cikm 2026/experiments"
CUTOFFS = [1, 3, 5, 10, 30, 50, 100]
B_BOOTSTRAP = 10000
ALPHA = 0.05
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


# ----- Load qrels -----
def load_qrels(path):
    qrels = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["query_id", "iteration", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )
    return qrels[["query_id", "doc_id", "relevance"]]


# ----- Load runs -----
def load_run(path, system_name):
    df = pd.read_csv(path)
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    # ir_measures expects: query_id, doc_id, score
    run = df[["qid", "docno", "score"]].rename(
        columns={"qid": "query_id", "docno": "doc_id"}
    )
    run["system"] = system_name
    return run


def colbert_run_from_biased(path):
    """Load ColBERT ranking from final_df_colbert_biased.csv.
    File only has biased_rank (no score), so score = -biased_rank.
    Same doc pool as SBR/IPSsimRF for apples-to-apples comparison.
    """
    df = pd.read_csv(path, lineterminator="\n")
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    df["score"] = -df["biased_rank"].astype(float)
    df = df.drop_duplicates(["qid", "docno"], keep="first")
    return df[["qid", "docno", "score"]].rename(
        columns={"qid": "query_id", "docno": "doc_id"}
    )


# ----- Per-query nDCG via ir_measures -----
def per_query_ndcg(run, qrels, cutoffs):
    """Return dict {cutoff: {qid: ndcg}}."""
    measures = [nDCG @ k for k in cutoffs]
    out = {k: {} for k in cutoffs}
    for r in iter_calc(measures, qrels, run):
        # r.measure has form 'nDCG@10'; extract k
        m_str = str(r.measure)
        k = int(m_str.split("@")[1])
        out[k][r.query_id] = r.value
    return out


def aligned_arrays(per_q_a, per_q_b):
    """Return two arrays of nDCG values aligned on shared qids."""
    qids = sorted(set(per_q_a.keys()) & set(per_q_b.keys()))
    a = np.array([per_q_a[q] for q in qids])
    b = np.array([per_q_b[q] for q in qids])
    return qids, a, b


# ----- Statistical tests -----
def paired_bootstrap_pvalue(a, b, n_boot=B_BOOTSTRAP, rng=None):
    """Two-sided paired bootstrap p-value for H0: mean(a) == mean(b)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    diffs = a - b
    obs_mean = diffs.mean()
    n = len(diffs)
    # Center under H0
    centered = diffs - obs_mean
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = centered[idx].mean()
    # Two-sided p-value
    p = (np.sum(np.abs(boot_means) >= abs(obs_mean)) + 1) / (n_boot + 1)
    return float(p)


def cohens_d_paired(a, b):
    """Paired Cohen's d = mean_diff / std_diff (using ddof=1)."""
    d = a - b
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


# ----- Main -----
def main():
    print("Loading qrels ...")
    qrels = load_qrels(QRELS_PATH)
    print(f"  {len(qrels)} judgments, {qrels['query_id'].nunique()} queries in qrels file")

    print("Loading runs ...")
    runs = {
        "ColBERT": colbert_run_from_biased(os.path.join(RANKINGS_DIR, "final_df_colbert_biased.csv")),
        "SBR": load_run(os.path.join(RANKINGS_DIR, "sbr_for_evaluation.csv"), "SBR"),
        "IPSsimRF": load_run(os.path.join(RANKINGS_DIR, "ipssimrf_for_evaluation.csv"), "IPSsimRF"),
    }
    for name, r in runs.items():
        print(f"  {name}: {len(r)} rows, {r['query_id'].nunique()} queries")

    print(f"Computing per-query nDCG @ {CUTOFFS} ...")
    per_q_all = {}
    for name, run in runs.items():
        per_q_all[name] = per_query_ndcg(run, qrels, CUTOFFS)

    # ----- Save per-query CSV -----
    rows = []
    for system, per_q_at_k in per_q_all.items():
        for k, qmap in per_q_at_k.items():
            for qid, val in qmap.items():
                rows.append({"qid": qid, "system": system, "cutoff": k, "ndcg": val})
    per_q_df = pd.DataFrame(rows)
    per_q_path = os.path.join(OUT_DIR, "a1_per_query_ndcg.csv")
    per_q_df.to_csv(per_q_path, index=False)
    print(f"Saved per-query nDCG: {per_q_path}")

    # ----- Pairwise tests -----
    pairs = [("ColBERT", "SBR"), ("ColBERT", "IPSsimRF"), ("SBR", "IPSsimRF")]
    n_tests = len(pairs) * len(CUTOFFS)
    bonf_alpha = ALPHA / n_tests

    results = []
    for k in CUTOFFS:
        for sys_a, sys_b in pairs:
            qids, a, b = aligned_arrays(per_q_all[sys_a][k], per_q_all[sys_b][k])
            t_stat, t_p = stats.ttest_rel(a, b)
            boot_p = paired_bootstrap_pvalue(a, b, n_boot=B_BOOTSTRAP, rng=np.random.default_rng(SEED + k))
            d = cohens_d_paired(a, b)
            results.append({
                "cutoff": k,
                "comparison": f"{sys_a} vs {sys_b}",
                "n_queries": len(qids),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "delta_a_minus_b": float((a - b).mean()),
                "t_pvalue": float(t_p),
                "bootstrap_pvalue": boot_p,
                "cohens_d": d,
                "bonferroni_significant_t": bool(t_p < bonf_alpha),
                "bonferroni_significant_bootstrap": bool(boot_p < bonf_alpha),
            })
    res_df = pd.DataFrame(results)
    res_path = os.path.join(OUT_DIR, "a1_significance_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"Saved significance results: {res_path}")
    print(f"Bonferroni alpha = {ALPHA} / {n_tests} = {bonf_alpha:.5f}")

    # ----- Pretty print -----
    print("\n========== Per-system mean nDCG ==========")
    means = per_q_df.groupby(["system", "cutoff"])["ndcg"].mean().unstack("cutoff")
    print(means.round(4).to_string())

    print("\n========== Pairwise tests ==========")
    fmt_cols = ["cutoff", "comparison", "delta_a_minus_b", "t_pvalue", "bootstrap_pvalue", "cohens_d",
                "bonferroni_significant_t", "bonferroni_significant_bootstrap"]
    print(res_df[fmt_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
