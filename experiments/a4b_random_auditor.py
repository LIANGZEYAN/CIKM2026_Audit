"""
A4b — Simulated random auditor (CIKM 2026 resubmission).

Replaces the real-click-derived anchor with N=3 uniformly random documents
sampled from each query's historical pool, then re-runs the full IPSsimRF
α-tuning protocol (RD-stability composite, lambda_range=(0.5, 3.0), 50 steps,
stability_weight=0.5, stability_cap=0.7, max_lambda=3.0) under the same 5-fold
CV harness used by a23_5fold_cv.py. The RD supervision is unchanged — what's
randomised is which docs the user "selected" to anchor semantic similarity.

Tests the AC's question: does the *human* signal carry information beyond the
structural choice of pool? If A4b ≈ A2, then any random doc-trio works as well
as user-selected anchors → user signal is structural; if A4b << A2, the audit
signal carries genuine pseudo-relevance information.

Reports mean ± SD over N_SEEDS=5 independent random-anchor draws (so the row
characterises the random *distribution*, not a single sample).

Pipeline (same as a23_5fold_cv.py):
  1. Load + dedupe historical pool
  2. PBM-simulate clicks (seed=505, total=1e6)
  3. Build RD over displayed audit-grid docs (from btdi_selection_results.csv)
  4. SimCSE embed every (qid, docno) text once
  5. For each random-anchor seed:
       a. Sample 3 random anchor docnos per qid from pool
       b. Compute random-anchored semantic_sim over full pool
       c. 5-fold CV: per fold, optimise α via RD-stability on training-fold
          displayed docs, evaluate held-out fold's full pool against TREC qrels
       d. Pool the 5 folds into per-seed pooled nDCG@k
  6. Aggregate over seeds: per-cutoff mean ± SD

Inputs (same as a23):
  /mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv
  /mnt/primary/Trec-llm/colbert_ipssim/all_selections_with_text.csv
  /mnt/primary/Trec-llm/colbert_ipssim/btdi_selection_results.csv
  /mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels
  SimCSE: princeton-nlp/sup-simcse-bert-base-uncased

Outputs:
  a4b_per_seed_per_fold.csv  — long: seed, fold, best_alpha, ndcg@k
  a4b_per_seed_pooled.csv    — seed, alpha_mean, alpha_sd, pooled_ndcg@k
  a4b_summary.csv            — cutoff, mean, sd, min, max over seeds
  a4b_table_row.tex          — LaTeX row for Table 1 (mean ± sd at each cutoff)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# ---- Config ----
HIST_PATH = "/mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv"
REAL_CLICKS_PATH = "/mnt/primary/Trec-llm/colbert_ipssim/all_selections_with_text.csv"
DISPLAY_PATH = "/mnt/primary/Trec-llm/colbert_ipssim/btdi_selection_results.csv"
QRELS_PATH = "/mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels"
OUT_DIR = "/mnt/primary/cikm 2026/experiments"
SIMCSE_MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"
HF_CACHE = "/mnt/primary/huggingface_cache"

TOP_K_ANCHOR = 3            # match real-click anchor count
SIM_SEED = 505              # PBM click simulation seed (matches a23/cttr)
TOTAL_CLICKS = 1_000_000
CUTOFFS = [1, 3, 5, 10, 30, 50, 100]
N_FOLDS = 5
SPLIT_SEED = 42

RANDOM_ANCHOR_SEEDS = [42, 43, 44, 45, 46]   # 5 seeds for the random-anchor draw

# RD-stability protocol (same as A2 / paper IPSsimRF tuning)
A2_RD_THRESHOLD = 1.5
A2_LAMBDA_RANGE = (0.5, 3.0)
A2_LAMBDA_STEPS = 50
A2_MAX_LAMBDA = 3.0
A2_STABILITY_WEIGHT = 0.5
A2_STABILITY_CAP = 0.7

PARAM_BETA = 0.0
PARAM_ALPHA_DIV = max(1.0, 100.0 / np.sqrt(TOTAL_CLICKS))  # = 1.0

os.environ["HF_HOME"] = HF_CACHE
os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# Pipeline helpers (copied verbatim from a23_5fold_cv.py for standalone use)
# =================================================================

def remove_duplicates_on_text(df, qid_col="qid", text_col="text", rank_col="rank"):
    df = df.copy()
    df["_idx"] = range(len(df))
    df.sort_values([qid_col, "_idx"], ascending=[True, True], inplace=True)
    parts = []
    for qid, sub in df.groupby(qid_col, group_keys=False):
        seen, keep = set(), []
        for _, row in sub.iterrows():
            t = str(row[text_col]).strip().lower()
            if t not in seen:
                seen.add(t)
                keep.append(row)
        parts.append(pd.DataFrame(keep))
    out = pd.concat(parts, ignore_index=True)
    out.sort_values("_idx", inplace=True)
    out.drop(columns=["_idx"], inplace=True)
    finals = []
    for qid, sub in out.groupby(qid_col, group_keys=False):
        sub = sub.copy()
        sub[rank_col] = range(len(sub))
        finals.append(sub)
    final = pd.concat(finals, ignore_index=True)
    final.sort_values([qid_col, rank_col], inplace=True)
    final.reset_index(drop=True, inplace=True)
    return final


def simulate_affine_clicks(df, total_clicks_target=TOTAL_CLICKS, seed=SIM_SEED):
    np.random.seed(seed)
    df = df.copy()

    def _mm(s):
        a, b = s.min(), s.max()
        return (s - a) / (b - a) if b > a else pd.Series([0.5] * len(s), index=s.index)
    df["normalized_score"] = df.groupby("qid")["score"].transform(_mm)
    df["pbm_alpha"] = df["rank"].apply(lambda k: 1.0 / (1.0 + np.log1p(k + 1)))
    df["pbm_beta"] = df["rank"].apply(lambda k: 0.1 / (1.0 + (k + 1)))
    df["click_probability"] = df["pbm_alpha"] * df["normalized_score"] + df["pbm_beta"]
    total_p = df["click_probability"].sum()
    qid_clicks = {}
    for qid, sub in df.groupby("qid"):
        share = sub["click_probability"].sum() / total_p
        qid_clicks[qid] = max(1, int(round(total_clicks_target * share)))
    counts = []
    for qid, sub in df.groupby("qid", sort=False):
        n = qid_clicks[qid]
        probs = sub["click_probability"].values
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        ks = np.random.multinomial(n, probs)
        counts.extend(ks.tolist())
    df["click_count"] = counts
    return df


def load_qrels():
    return pd.read_csv(
        QRELS_PATH, sep=r"\s+", header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )[["query_id", "doc_id", "relevance"]]


def build_random_anchor_map(hist_df, top_k, anchor_seed):
    """A4b anchor: per qid, sample top_k random docnos uniformly without replacement."""
    rng = np.random.RandomState(anchor_seed)
    out = {}
    for qid, sub in hist_df.groupby("qid"):
        docnos = sub["docno"].astype(str).tolist()
        k = min(top_k, len(docnos))
        idx = rng.choice(len(docnos), size=k, replace=False)
        out[str(qid)] = [docnos[i] for i in idx]
    return out


def embed_full_pool(df, text_col="text", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SimCSE on device: {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
    model = AutoModel.from_pretrained(SIMCSE_MODEL).to(device).eval()

    texts = df[text_col].fillna("").astype(str).tolist()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            embs.append(out.pooler_output.cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"    embedded {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
    embs = np.concatenate(embs, axis=0)
    qids = df["qid"].astype(str).tolist()
    docs = df["docno"].astype(str).tolist()
    return {(qids[i], docs[i]): embs[i] for i in range(len(embs))}


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_semantic_sim(df, anchor_map, emb_lookup):
    sims = np.zeros(len(df))
    for i, (qid, docno) in enumerate(zip(df["qid"].astype(str).values,
                                         df["docno"].astype(str).values)):
        target = emb_lookup.get((qid, docno))
        anchors = anchor_map.get(qid, [])
        if target is None or not anchors:
            sims[i] = 0.0
            continue
        ss = []
        for a in anchors:
            if a == docno:
                continue
            a_emb = emb_lookup.get((qid, a))
            if a_emb is None:
                continue
            ss.append(cosine_sim(target, a_emb))
        sims[i] = float(np.mean(ss)) if ss else 0.0
    return sims


def compute_rd_on_displayed(df_simulated_displayed, real_clicks_counts):
    df = df_simulated_displayed.copy()
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    rc = real_clicks_counts.rename(columns={"click_count": "real_clicks"}).copy()
    rc["qid"] = rc["qid"].astype(str)
    rc["docno"] = rc["docno"].astype(str)
    df = df.merge(rc[["qid", "docno", "real_clicks"]], on=["qid", "docno"], how="left")
    df["real_clicks"] = df["real_clicks"].fillna(0)
    df["real_ctr"] = df.groupby("qid")["real_clicks"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x * 0
    )
    df["ctr_ratio"] = df.groupby("qid")["ctr"].transform(
        lambda x: x / (x.sum() + 1e-10)
    )
    df["rd"] = df["real_ctr"] / (df["ctr_ratio"] + 1e-10)
    return df


def optimize_lambda_rd_stability(df_displayed_train, sim_col):
    df = df_displayed_train.copy()
    if "doc_alpha" not in df.columns:
        df["doc_alpha"] = 1.0
    if "doc_beta" not in df.columns:
        df["doc_beta"] = 0.0

    def _mm(s):
        mn, mx = s.min(), s.max()
        if mx > mn:
            return (s - mn) / (mx - mn)
        return pd.Series([0.5] * len(s), index=s.index)
    df["normalized_ctr"] = df.groupby("qid")["ctr"].transform(_mm)

    df["is_high_rd"] = df["rd"] > A2_RD_THRESHOLD

    lambda_values = np.linspace(A2_LAMBDA_RANGE[0], A2_LAMBDA_RANGE[1], A2_LAMBDA_STEPS)
    rows = []
    for lam in lambda_values:
        adj = (df["normalized_ctr"] - df["doc_beta"] + lam * df[sim_col]) / df["doc_alpha"]
        df["_adj"] = adj
        high_low_diff = 0.0
        rank_changes_abs = []
        for qid, qd in df.groupby("qid"):
            if len(qd) < 2:
                continue
            qd = qd.copy()
            qd["orig_rank"] = qd["normalized_ctr"].rank(ascending=False)
            qd["new_rank"] = qd["_adj"].rank(ascending=False)
            qd["rank_change"] = qd["orig_rank"] - qd["new_rank"]
            high = qd[qd["is_high_rd"]]
            low = qd[~qd["is_high_rd"]]
            if len(high) > 0 and len(low) > 0:
                high_low_diff += high["rank_change"].mean() - low["rank_change"].mean()
            rank_changes_abs.extend(qd["rank_change"].abs().tolist())

        avg_abs_change = float(np.mean(rank_changes_abs)) if rank_changes_abs else 0.0
        raw_stability = 1.0 / (1.0 + avg_abs_change)
        capped_stability = min(raw_stability, A2_STABILITY_CAP)
        rows.append({"lambda": float(lam),
                     "high_low_diff": float(high_low_diff),
                     "raw_stability": raw_stability,
                     "capped_stability": capped_stability})

    rdf = pd.DataFrame(rows)
    max_diff = rdf["high_low_diff"].max()
    rdf["norm_diff"] = rdf["high_low_diff"] / max_diff if max_diff > 0 else 0.0
    rdf["score"] = (1 - A2_STABILITY_WEIGHT) * rdf["norm_diff"] \
                   + A2_STABILITY_WEIGHT * rdf["capped_stability"]

    valid = rdf[rdf["lambda"] <= A2_MAX_LAMBDA]
    if len(valid) == 0:
        valid = rdf
    best_idx = valid["score"].idxmax()
    best_lambda = float(valid.loc[best_idx, "lambda"])
    return best_lambda


def evaluate_held_out(df_pool_test, sim_col, alpha, qrels, cutoffs=CUTOFFS):
    """Apply alpha to held-out queries' full pool, return per-cutoff nDCG dict + run df."""
    from ir_measures import nDCG, calc_aggregate
    df = df_pool_test.copy()
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)

    def _mm(s):
        mn, mx = s.min(), s.max()
        if mx > mn:
            return (s - mn) / (mx - mn)
        return pd.Series([0.5] * len(s), index=s.index)
    df["normalized_ctr"] = df.groupby("qid")["ctr"].transform(_mm)

    score = (df["normalized_ctr"] - PARAM_BETA + alpha * df[sim_col]) / PARAM_ALPHA_DIV
    run = pd.DataFrame({
        "query_id": df["qid"].values,
        "doc_id": df["docno"].values,
        "score": score.values,
    })
    test_qids_str = set(df["qid"].unique())
    qrels_test = qrels[qrels["query_id"].isin(test_qids_str)]
    measures = [nDCG @ k for k in cutoffs]
    res = calc_aggregate(measures, qrels_test, run)
    out = {}
    for m, v in res.items():
        m_str = str(m)
        k = int(m_str.split("@")[1])
        out[k] = float(v)
    return out, run


def make_folds(qids, n_folds=N_FOLDS, seed=SPLIT_SEED):
    qids = sorted(set(int(q) for q in qids))
    rng = np.random.RandomState(seed)
    arr = np.array(qids)
    rng.shuffle(arr)
    folds = np.array_split(arr, n_folds)
    return [[int(q) for q in fold] for fold in folds]


# =================================================================
# Main
# =================================================================

def main():
    print("=" * 88)
    print("A4b — Simulated random auditor (CIKM 2026 resubmission)")
    print(f"     {len(RANDOM_ANCHOR_SEEDS)} random-anchor seeds × 5-fold CV × RD-stability tune")
    print("=" * 88)

    print("\nStep 1/6: Load + dedupe historical pool ...", flush=True)
    hist_raw = pd.read_csv(HIST_PATH, lineterminator="\n")
    hist_raw["qid"] = hist_raw["qid"].astype(int)
    hist_raw["docno"] = hist_raw["docno"].astype(str)
    hist_raw["score"] = hist_raw["score"].astype(float)
    hist = remove_duplicates_on_text(hist_raw, qid_col="qid", text_col="text", rank_col="rank")
    print(f"  After dedup: {len(hist)} rows, {hist['qid'].nunique()} qids, "
          f"docs/qid mean={hist.groupby('qid').size().mean():.1f}")

    print("\nStep 2/6: PBM-simulate clicks (seed=505, total=1e6) ...", flush=True)
    sim = simulate_affine_clicks(hist[["qid", "query", "docno", "text", "score", "rank"]].copy())
    sim["ctr"] = sim.groupby("qid")["click_count"].transform(lambda x: x / x.sum())
    sim["biased_rank"] = sim.groupby("qid")["ctr"].rank(method="first", ascending=False).astype(int)
    print(f"  Simulated total: {int(sim['click_count'].sum()):,}")

    print("\nStep 3/6: Load real clicks + build RD over displayed grid ...", flush=True)
    click_df = pd.read_csv(REAL_CLICKS_PATH, dtype={"qid": str, "docno": str})
    real_clicks_counts = click_df.groupby(["qid", "docno"]).size().reset_index(name="click_count")
    print(f"  Real-click events: {int(real_clicks_counts['click_count'].sum()):,} over "
          f"{real_clicks_counts['qid'].nunique()} qids")

    display_df = pd.read_csv(DISPLAY_PATH)
    display_df["qid"] = display_df["qid"].astype(str)
    display_df["docno"] = display_df["docno"].astype(str)
    sim["qid_s"] = sim["qid"].astype(str)
    sim_disp = sim.merge(display_df[["qid", "docno"]],
                         left_on=["qid_s", "docno"], right_on=["qid", "docno"],
                         how="inner", suffixes=("", "_disp"))
    sim_disp = sim_disp.drop(columns=["qid_s", "qid_disp"], errors="ignore")
    sim_disp["qid"] = sim_disp["qid"].astype(str)
    sim.drop(columns=["qid_s"], inplace=True)
    disp_with_rd = compute_rd_on_displayed(sim_disp, real_clicks_counts)
    print(f"  Displayed grid: {len(disp_with_rd)} rows × "
          f"{disp_with_rd['qid'].nunique()} qids")
    print(f"  RD: mean={disp_with_rd['rd'].mean():.4f}, "
          f"high(>1.5)={(disp_with_rd['rd']>1.5).sum()}/{len(disp_with_rd)}")

    print("\nStep 4/6: SimCSE embed full pool ...", flush=True)
    sim_for_embed = sim[["qid", "docno", "text"]].copy()
    sim_for_embed["qid"] = sim_for_embed["qid"].astype(str)
    emb_lookup = embed_full_pool(sim_for_embed, text_col="text", batch_size=32)
    print(f"  Embedded {len(emb_lookup)} docs")

    print("\nStep 5/6: 5-fold splits ...", flush=True)
    all_qids = sorted(sim["qid"].unique().tolist())
    folds = make_folds(all_qids, n_folds=N_FOLDS, seed=SPLIT_SEED)
    qrels = load_qrels()

    hist_for_anchor = sim[["qid", "docno", "score"]].copy()
    hist_for_anchor["qid"] = hist_for_anchor["qid"].astype(str)

    print("\nStep 6/6: Loop over random-anchor seeds × folds ...", flush=True)
    per_seed_per_fold_rows = []
    per_seed_pooled_rows = []

    for s_idx, anchor_seed in enumerate(RANDOM_ANCHOR_SEEDS):
        print(f"\n  ===== Random-anchor seed {anchor_seed} ({s_idx+1}/{len(RANDOM_ANCHOR_SEEDS)}) =====")
        # Build random anchors
        random_anchor_map = build_random_anchor_map(hist_for_anchor, top_k=TOP_K_ANCHOR,
                                                    anchor_seed=anchor_seed)
        # Compute random-anchored sim over full pool
        sim["qid_s"] = sim["qid"].astype(str)
        sim_str = sim[["qid_s", "docno"]].rename(columns={"qid_s": "qid"})
        sim_random_col = compute_semantic_sim(sim_str, random_anchor_map, emb_lookup)
        sim["semantic_sim_random"] = sim_random_col
        sim.drop(columns=["qid_s"], inplace=True)
        # Project sim_random onto displayed-grid for RD-stability tuning
        disp_with_rd_seed = disp_with_rd.merge(
            sim[["qid", "docno", "semantic_sim_random"]].astype({"qid": str, "docno": str}),
            on=["qid", "docno"], how="left", suffixes=("", "_full"))
        disp_with_rd_seed["semantic_sim_random"] = disp_with_rd_seed["semantic_sim_random"].fillna(0.0)
        print(f"    sim_random over full pool: mean={sim_random_col.mean():.4f} "
              f"(reference: A2 click-anchor mean ≈ 0.58)")

        per_fold_runs = []
        per_fold_alphas = []
        for fold_idx, test_qids in enumerate(folds):
            train_qids = [q for fi, f in enumerate(folds) for q in f if fi != fold_idx]
            disp_train = disp_with_rd_seed[disp_with_rd_seed["qid"].astype(int).isin(train_qids)]
            best_alpha = optimize_lambda_rd_stability(disp_train, sim_col="semantic_sim_random")
            pool_test = sim[sim["qid"].isin(test_qids)]
            metrics, run = evaluate_held_out(pool_test, sim_col="semantic_sim_random",
                                             alpha=best_alpha, qrels=qrels)
            per_fold_runs.append(run)
            per_fold_alphas.append(best_alpha)
            row = {"anchor_seed": anchor_seed, "fold": fold_idx,
                   "n_train": len(train_qids), "n_test": len(test_qids),
                   "best_alpha": best_alpha}
            for k in CUTOFFS:
                row[f"ndcg{k}"] = metrics[k]
            per_seed_per_fold_rows.append(row)
            print(f"    fold {fold_idx}: α={best_alpha:.4f}, " +
                  " ".join(f"@{k}={metrics[k]:.4f}" for k in CUTOFFS))

        # Pool 5 folds for this seed
        from ir_measures import nDCG, calc_aggregate
        runs_concat = pd.concat(per_fold_runs, ignore_index=True)
        measures = [nDCG @ k for k in CUTOFFS]
        res = calc_aggregate(measures, qrels, runs_concat)
        rec = {"anchor_seed": anchor_seed,
               "alpha_mean": float(np.mean(per_fold_alphas)),
               "alpha_sd": float(np.std(per_fold_alphas, ddof=1)) if len(per_fold_alphas) > 1 else 0.0,
               "alpha_min": float(np.min(per_fold_alphas)),
               "alpha_max": float(np.max(per_fold_alphas))}
        for m, v in res.items():
            k = int(str(m).split("@")[1])
            rec[f"ndcg{k}"] = float(v)
        per_seed_pooled_rows.append(rec)
        print(f"    POOLED: α={rec['alpha_mean']:.3f}±{rec['alpha_sd']:.3f}, " +
              " ".join(f"@{k}={rec[f'ndcg{k}']:.4f}" for k in CUTOFFS))

    # ---- Save per-fold and per-seed-pooled ----
    pf_df = pd.DataFrame(per_seed_per_fold_rows)
    pf_df.to_csv(os.path.join(OUT_DIR, "a4b_per_seed_per_fold.csv"), index=False)
    ps_df = pd.DataFrame(per_seed_pooled_rows)
    ps_df.to_csv(os.path.join(OUT_DIR, "a4b_per_seed_pooled.csv"), index=False)

    # ---- Aggregate across seeds: mean ± SD per cutoff ----
    summary_rows = []
    for k in CUTOFFS:
        vals = ps_df[f"ndcg{k}"].values
        summary_rows.append({
            "cutoff": k,
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "a4b_summary.csv"), index=False)

    alpha_overall_mean = float(ps_df["alpha_mean"].mean())
    alpha_overall_sd = float(ps_df["alpha_mean"].std(ddof=1))

    # ---- Print final summary ----
    print("\n" + "=" * 88)
    print("A4b SUMMARY (across 5 random-anchor seeds, 5-fold CV)")
    print("=" * 88)
    print(f"α (mean of per-seed alpha_means): {alpha_overall_mean:.4f} ± {alpha_overall_sd:.4f}")
    header = f"{'cutoff':<8} {'mean':<8} {'sd':<8} {'min':<8} {'max':<8}"
    print(header); print("-" * len(header))
    for r in summary_rows:
        print(f"@{r['cutoff']:<7} {r['mean']:.4f}   {r['sd']:.4f}   {r['min']:.4f}   {r['max']:.4f}")

    print("\nReference rows (from a23_pooled.csv):")
    print("  A2 (real-click anchor, RD-stability):  α=2.867, "
          "@1=0.6950 @3=0.6849 @5=0.6611 @10=0.6236 @30=0.5344 @50=0.4786 @100=0.3899")
    print("  A3 (ColBERT anchor, nDCG-grid):          α=2.940, "
          "@1=0.6069 @3=0.6503 @5=0.6378 @10=0.6049 @30=0.5238 @50=0.4685 @100=0.3829")

    # ---- LaTeX row ----
    tex_path = os.path.join(OUT_DIR, "a4b_table_row.tex")
    with open(tex_path, "w") as f:
        f.write("% A4b row for Table 1 (5 random-anchor seeds × 5-fold CV; mean ± sd over seeds)\n")
        f.write("IPSsimRF (random auditor) & "
                + " & ".join(
                    f"{r['mean']:.4f}\\,$\\pm$\\,{r['sd']:.4f}"
                    for r in summary_rows)
                + " \\\\\n")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'a4b_per_seed_per_fold.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'a4b_per_seed_pooled.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'a4b_summary.csv')}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
