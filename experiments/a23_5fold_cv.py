"""
A2 + A3 — 5-fold cross-validation for α (CIKM 2026 resubmission).

Combines two reviewer requests under a single 5-fold CV harness so the
comparison is apples-to-apples.

  A2 (AC mRFk + oJ4e B7) — 5-fold CV of original IPSsimRF.
    Reproduces the RD-based α-tuning protocol from cttr.ipynb cell 9
    (`optimize_lambda_simple`): per-fold, tune α on training queries'
    displayed audit-grid docs using a RD-stability composite objective.
    Hyperparameters held at the values that produced paper's α=2.6429
    (rd_threshold=1.5, lambda_range=(0.5, 3.0), 50 steps, max_lambda=3.0,
    stability_weight=0.5, stability_cap=0.7). Removes the in-sample
    optimism the AC flagged.

  A3 (AC, run-first) — no-audit α baseline.
    Same score formula and 5-fold split, but α is tuned against TREC
    qrel-derived nDCG@10 (no user signal), and the semantic-similarity
    anchor is the per-query ColBERT historical-rank top-3 (no user signal).
    Tests AC's question: would tuning α with a non-audit supervision +
    non-audit anchor reach IPSsimRF's nDCG?

Both variants use the same scoring formula matched to version_2 cell 9:
    score = (norm_ctr − β + α · sim) / max(1.0, 100/√(total_clicks))
with β=0 and total_clicks=1e6 (so the divisor is exactly 1.0 here).

Pipeline replicates ipssim_version_2.ipynb cells 5--9 + cttr.ipynb cell 9:
  1. Load + dedupe ColBERT historical pool (53 qids, ~5300 docs)
  2. PBM-simulate clicks (seed=505, total=1e6) → simulated `ctr`, `biased_rank`
  3. Load real clicks from all_selections_with_text.csv → top-3 per-qid map
  4. Build ColBERT top-3 anchor map (per-qid by historical_score)
  5. Single-pass SimCSE embed of every (qid, docno) text in the full pool
  6. Compute two `semantic_sim_*` columns over the full pool:
       semantic_sim_clicks  — anchor = real-click top-3 per qid (used for A2 + IPSsimRF eval)
       semantic_sim_colbert — anchor = ColBERT top-3 per qid (used for A3)
  7. Build RD per (qid, displayed-docno) from btdi_selection_results.csv
  8. 5-fold split queries (sorted asc by qid, np.random.RandomState(42).shuffle,
     np.array_split) → ~11/11/11/10/10
  9. Per fold, both variants:
       A2: optimize_lambda_simple on training-fold displayed docs via RD-stability
       A3: grid α∈[0,5] step 0.1 on training-fold full pool nDCG@10 mean
       Apply best α to held-out fold's full pool and evaluate nDCG@k.
 10. Pool: each of the 53 queries was held-out exactly once → concatenate
     53 per-query rankings (each scored with its own fold's α) and run a
     single calc_aggregate for the pooled per-cutoff nDCG.

Inputs:
  /mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv
  /mnt/primary/Trec-llm/colbert_ipssim/all_selections_with_text.csv
  /mnt/primary/Trec-llm/colbert_ipssim/btdi_selection_results.csv
  /mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels
  SimCSE: princeton-nlp/sup-simcse-bert-base-uncased

Outputs (in /mnt/primary/cikm 2026/experiments/):
  a23_per_fold.csv     — long format: variant, fold, n_train, n_test, best_alpha, ndcg@k
  a23_pooled.csv       — variant × cutoff pooled nDCG; α stats
  a23_table_rows.tex   — LaTeX rows for Table 1 (A2 cv-tuned, A3 no-audit cv)
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

TOP_K_ANCHOR_CLICKS = 3   # version_2.ipynb cell 7
TOP_K_ANCHOR_COLBERT = 3  # match for parity in A3
SIM_SEED = 505
TOTAL_CLICKS = 1_000_000
CUTOFFS = [1, 3, 5, 10, 30, 50, 100]
N_FOLDS = 5
SPLIT_SEED = 42

# A2 — exact cttr.ipynb cell 9 hyperparameters
A2_RD_THRESHOLD = 1.5
A2_LAMBDA_RANGE = (0.5, 3.0)
A2_LAMBDA_STEPS = 50
A2_MAX_LAMBDA = 3.0
A2_STABILITY_WEIGHT = 0.5
A2_STABILITY_CAP = 0.7

# A3 — clean nDCG grid
A3_ALPHA_GRID = np.round(np.arange(0.0, 5.01, 0.1), 2).tolist()

# Score formula constants (version_2 cell 9)
PARAM_BETA = 0.0
PARAM_ALPHA_DIV = max(1.0, 100.0 / np.sqrt(TOTAL_CLICKS))  # = 1.0

os.environ["HF_HOME"] = HF_CACHE
os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# Pipeline cells (mirroring version_2.ipynb / cttr.ipynb)
# =================================================================

def remove_duplicates_on_text(df, qid_col="qid", text_col="text", rank_col="rank"):
    """Cell 1 of cttr/version_2: dedupe by lowercased+stripped text within each qid; reassign rank."""
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
    """Cell 5 of v2: PBM affine click simulation, deterministic."""
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


def build_click_anchor_map(real_clicks_csv_path, top_k=TOP_K_ANCHOR_CLICKS):
    """version_2 cell 7: per-qid top-K most-clicked docnos from all_selections_with_text.csv."""
    click_df = pd.read_csv(real_clicks_csv_path, dtype={"qid": str, "docno": str})
    click_counts = click_df.groupby(["qid", "docno"]).size().reset_index(name="click_count")
    out = {}
    for qid, group in click_counts.groupby("qid"):
        out[str(qid)] = group.nlargest(top_k, "click_count")["docno"].astype(str).tolist()
    return out, click_counts


def build_colbert_anchor_map(hist_df, top_k=TOP_K_ANCHOR_COLBERT):
    """A3 anchor: per-qid top-K by ColBERT historical score (no user signal)."""
    out = {}
    for qid, sub in hist_df.groupby("qid"):
        top = sub.nlargest(top_k, "score")["docno"].astype(str).tolist()
        out[str(qid)] = top
    return out


# =================================================================
# SimCSE
# =================================================================

def embed_full_pool(df, text_col="text", batch_size=32):
    """Embed every (qid, docno) text once. Returns dict (qid, docno) -> np.ndarray."""
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
    lookup = {(qids[i], docs[i]): embs[i] for i in range(len(embs))}
    return lookup


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_semantic_sim(df, anchor_map, emb_lookup):
    """For each row, compute mean cosine sim to its qid's anchors (excluding self)."""
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


# =================================================================
# RD (cttr.ipynb cell 6: analyze_rd_distribution, simplified)
# =================================================================

def compute_rd_on_displayed(df_simulated_displayed, real_clicks_counts):
    """
    Replicates analyze_rd_distribution's RD computation, restricted to the
    9-doc audit grid per query.

    df_simulated_displayed: rows = (qid, docno) ∈ btdi_selection_results.csv,
                            with simulated `ctr` from PBM and pool docs joined.
    real_clicks_counts:     DataFrame with (qid, docno, click_count) from
                            all_selections_with_text.csv groupby.

    Returns df with added columns: real_clicks, real_ctr, ctr_ratio, rd.
    """
    df = df_simulated_displayed.copy()
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    rc = real_clicks_counts.rename(columns={"click_count": "real_clicks"}).copy()
    rc["qid"] = rc["qid"].astype(str)
    rc["docno"] = rc["docno"].astype(str)
    df = df.merge(rc[["qid", "docno", "real_clicks"]], on=["qid", "docno"], how="left")
    df["real_clicks"] = df["real_clicks"].fillna(0)
    # real_ctr and ctr_ratio normalised within DISPLAYED docs (cttr.ipynb cell 6 lines 57, 62)
    df["real_ctr"] = df.groupby("qid")["real_clicks"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x * 0
    )
    df["ctr_ratio"] = df.groupby("qid")["ctr"].transform(
        lambda x: x / (x.sum() + 1e-10)
    )
    df["rd"] = df["real_ctr"] / (df["ctr_ratio"] + 1e-10)
    return df


# =================================================================
# Objective functions
# =================================================================

def optimize_lambda_rd_stability(df_displayed_train, sim_col):
    """
    Replica of cttr.ipynb cell 6 `optimize_lambda_simple` — restricted to the
    queries in the training fold's displayed-grid docs.

    Returns: best_lambda, scoring_history (list of dicts).
    """
    df = df_displayed_train.copy()
    if "doc_alpha" not in df.columns:
        df["doc_alpha"] = 1.0
    if "doc_beta" not in df.columns:
        df["doc_beta"] = 0.0

    # normalized_ctr (per-qid min-max over displayed docs of simulated ctr)
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
    return best_lambda, rdf


def optimize_alpha_ndcg(df_pool_train, sim_col, qrels):
    """
    A3 objective: grid α∈[0,5] step 0.1, maximise mean per-query nDCG@10 over
    the training-fold full-pool docs, evaluated against TREC qrels.

    Returns: best_alpha, scoring_history (list of dicts).
    """
    from ir_measures import nDCG, calc_aggregate
    df = df_pool_train.copy()
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)

    def _mm(s):
        mn, mx = s.min(), s.max()
        if mx > mn:
            return (s - mn) / (mx - mn)
        return pd.Series([0.5] * len(s), index=s.index)
    df["normalized_ctr"] = df.groupby("qid")["ctr"].transform(_mm)

    rows = []
    for alpha in A3_ALPHA_GRID:
        score = (df["normalized_ctr"] - PARAM_BETA + alpha * df[sim_col]) / PARAM_ALPHA_DIV
        run = pd.DataFrame({
            "query_id": df["qid"].values,
            "doc_id": df["docno"].values,
            "score": score.values,
        })
        res = calc_aggregate([nDCG @ 10], qrels, run)
        ndcg10 = float(list(res.values())[0])
        rows.append({"alpha": float(alpha), "ndcg10": ndcg10})

    rdf = pd.DataFrame(rows)
    best_idx = rdf["ndcg10"].idxmax()
    return float(rdf.loc[best_idx, "alpha"]), rdf


# =================================================================
# Held-out evaluation
# =================================================================

def evaluate_held_out(df_pool_test, sim_col, alpha, qrels, cutoffs=CUTOFFS):
    """Apply alpha to held-out queries' full pool, compute nDCG@k. Returns dict.

    NB: ir_measures `calc_aggregate` averages over the *qrels* universe of qids,
    so we must restrict the qrels to only the test fold's qids to get a per-fold
    mean (otherwise it's diluted by 42 missing-qid zeroes)."""
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
    print("A2 + A3 — 5-fold CV for α (CIKM 2026 resubmission)")
    print("=" * 88)

    print("\nStep 1/9: Load + dedupe historical pool ...", flush=True)
    hist_raw = pd.read_csv(HIST_PATH, lineterminator="\n")
    hist_raw["qid"] = hist_raw["qid"].astype(int)
    hist_raw["docno"] = hist_raw["docno"].astype(str)
    hist_raw["score"] = hist_raw["score"].astype(float)
    hist = remove_duplicates_on_text(hist_raw, qid_col="qid", text_col="text", rank_col="rank")
    print(f"  After dedup: {len(hist)} rows, {hist['qid'].nunique()} qids, "
          f"docs/qid mean={hist.groupby('qid').size().mean():.1f}")

    print("\nStep 2/9: PBM-simulate clicks (seed=505, total=1e6) ...", flush=True)
    sim = simulate_affine_clicks(
        hist[["qid", "query", "docno", "text", "score", "rank"]].copy()
    )
    sim["ctr"] = sim.groupby("qid")["click_count"].transform(lambda x: x / x.sum())
    sim["biased_rank"] = sim.groupby("qid")["ctr"].rank(method="first", ascending=False).astype(int)
    print(f"  Simulated clicks total: {int(sim['click_count'].sum()):,}")

    print("\nStep 3/9: Build click-anchor map (top-3 by real_clicks) ...", flush=True)
    click_anchor_map, real_clicks_counts = build_click_anchor_map(REAL_CLICKS_PATH,
                                                                  top_k=TOP_K_ANCHOR_CLICKS)
    print(f"  Real-click anchor map: {len(click_anchor_map)} qids, "
          f"event total={int(real_clicks_counts['click_count'].sum()):,}")

    print("\nStep 4/9: Build ColBERT-anchor map (top-3 by historical score) ...", flush=True)
    hist_for_anchor = sim[["qid", "docno", "score"]].copy()
    hist_for_anchor["qid"] = hist_for_anchor["qid"].astype(str)
    colbert_anchor_map = build_colbert_anchor_map(hist_for_anchor, top_k=TOP_K_ANCHOR_COLBERT)
    print(f"  ColBERT anchor map: {len(colbert_anchor_map)} qids")

    print("\nStep 5/9: SimCSE embed every (qid, docno) text in full pool ...", flush=True)
    sim_for_embed = sim[["qid", "docno", "text"]].copy()
    sim_for_embed["qid"] = sim_for_embed["qid"].astype(str)
    emb_lookup = embed_full_pool(sim_for_embed, text_col="text", batch_size=32)
    print(f"  Embedded {len(emb_lookup)} docs")

    print("\nStep 6/9: Compute semantic_sim_clicks and semantic_sim_colbert ...", flush=True)
    sim["qid_str"] = sim["qid"].astype(str)
    sim_str = sim[["qid_str", "docno"]].rename(columns={"qid_str": "qid"})
    sim["semantic_sim_clicks"] = compute_semantic_sim(sim_str, click_anchor_map, emb_lookup)
    sim["semantic_sim_colbert"] = compute_semantic_sim(sim_str, colbert_anchor_map, emb_lookup)
    sim.drop(columns=["qid_str"], inplace=True)
    print(f"  click-anchored sim: mean={sim['semantic_sim_clicks'].mean():.4f}, "
          f"colbert-anchored sim: mean={sim['semantic_sim_colbert'].mean():.4f}")

    print("\nStep 7/9: Build RD over displayed audit-grid docs ...", flush=True)
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
    print(f"  Displayed grid: {len(disp_with_rd)} rows over "
          f"{disp_with_rd['qid'].nunique()} qids "
          f"(should be ~9 docs × 53 qids = 477)")
    print(f"  RD stats: mean={disp_with_rd['rd'].mean():.4f}, "
          f"median={disp_with_rd['rd'].median():.4f}, "
          f"high(>1.5)={(disp_with_rd['rd']>1.5).sum()}/{len(disp_with_rd)}")

    print("\nStep 8/9: Make 5 folds ...", flush=True)
    all_qids = sorted(sim["qid"].unique().tolist())
    folds = make_folds(all_qids, n_folds=N_FOLDS, seed=SPLIT_SEED)
    for i, f in enumerate(folds):
        print(f"  Fold {i}: n={len(f)}, qids={f[:3]}...{f[-3:] if len(f)>3 else ''}")

    qrels = load_qrels()
    print(f"  Qrels: {qrels['query_id'].nunique()} unique qids, {len(qrels)} judgements")

    print("\nStep 9/9: 5-fold CV for A2 (RD-stability) and A3 (nDCG grid) ...", flush=True)

    per_fold_rows = []
    pooled_runs = {"A2": [], "A3": []}
    pooled_alphas = {"A2": [], "A3": []}

    for fold_idx, test_qids in enumerate(folds):
        train_qids = [q for f_idx, f in enumerate(folds) for q in f if f_idx != fold_idx]
        print(f"\n  --- Fold {fold_idx}: train={len(train_qids)}, test={len(test_qids)} ---")

        # ----- A2: RD-stability on training-fold displayed docs -----
        disp_train = disp_with_rd[disp_with_rd["qid"].astype(int).isin(train_qids)]
        best_a2, _ = optimize_lambda_rd_stability(
            disp_train, sim_col="semantic_sim_clicks"
        )
        print(f"    A2 best α (RD-stability on train) = {best_a2:.4f}")

        # ----- A3: nDCG@10 grid on training-fold full pool -----
        pool_train = sim[sim["qid"].isin(train_qids)]
        best_a3, _ = optimize_alpha_ndcg(
            pool_train, sim_col="semantic_sim_colbert", qrels=qrels
        )
        print(f"    A3 best α (nDCG@10 grid on train, ColBERT anchor) = {best_a3:.4f}")

        # ----- Apply to held-out fold -----
        pool_test = sim[sim["qid"].isin(test_qids)]
        a2_metrics, a2_run = evaluate_held_out(
            pool_test, sim_col="semantic_sim_clicks", alpha=best_a2, qrels=qrels
        )
        a3_metrics, a3_run = evaluate_held_out(
            pool_test, sim_col="semantic_sim_colbert", alpha=best_a3, qrels=qrels
        )

        n_train, n_test = len(train_qids), len(test_qids)
        a2_row = {"variant": "A2", "fold": fold_idx,
                  "n_train": n_train, "n_test": n_test, "best_alpha": best_a2}
        a3_row = {"variant": "A3", "fold": fold_idx,
                  "n_train": n_train, "n_test": n_test, "best_alpha": best_a3}
        for k in CUTOFFS:
            a2_row[f"ndcg{k}"] = a2_metrics[k]
            a3_row[f"ndcg{k}"] = a3_metrics[k]
        per_fold_rows.append(a2_row)
        per_fold_rows.append(a3_row)
        pooled_runs["A2"].append(a2_run)
        pooled_runs["A3"].append(a3_run)
        pooled_alphas["A2"].append(best_a2)
        pooled_alphas["A3"].append(best_a3)

        print(f"    A2 held-out: " + " ".join(f"@{k}={a2_metrics[k]:.4f}" for k in CUTOFFS))
        print(f"    A3 held-out: " + " ".join(f"@{k}={a3_metrics[k]:.4f}" for k in CUTOFFS))

    # ---- Save per-fold ----
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(os.path.join(OUT_DIR, "a23_per_fold.csv"), index=False)

    # ---- Pooled (concat all 53 query × cutoff) ----
    from ir_measures import nDCG, calc_aggregate
    pooled_rows = []
    for variant in ["A2", "A3"]:
        runs_concat = pd.concat(pooled_runs[variant], ignore_index=True)
        measures = [nDCG @ k for k in CUTOFFS]
        res = calc_aggregate(measures, qrels, runs_concat)
        rec = {"variant": variant,
               "alpha_mean": float(np.mean(pooled_alphas[variant])),
               "alpha_sd": float(np.std(pooled_alphas[variant], ddof=1)),
               "alpha_min": float(np.min(pooled_alphas[variant])),
               "alpha_max": float(np.max(pooled_alphas[variant]))}
        for m, v in res.items():
            k = int(str(m).split("@")[1])
            rec[f"ndcg{k}"] = float(v)
        pooled_rows.append(rec)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(os.path.join(OUT_DIR, "a23_pooled.csv"), index=False)

    # ---- Print summary ----
    print("\n" + "=" * 88)
    print("POOLED HELD-OUT RESULTS (5-fold CV; each query held-out once)")
    print("=" * 88)
    header = f"{'Variant':<8} {'α (mean±sd)':<14} {'α range':<14} " \
             + " ".join(f"@{k:<5}" for k in CUTOFFS)
    print(header)
    print("-" * len(header))
    for r in pooled_rows:
        a_str = f"{r['alpha_mean']:.3f}±{r['alpha_sd']:.3f}"
        r_str = f"[{r['alpha_min']:.2f},{r['alpha_max']:.2f}]"
        cuts = " ".join(f"{r[f'ndcg{k}']:.4f}" for k in CUTOFFS)
        print(f"{r['variant']:<8} {a_str:<14} {r_str:<14} {cuts}")

    print("\nReference (paper Table 1, IPSsimRF α=2.6429 in-sample):")
    print(f"{'IPSsimRF':<8} {'2.6429':<14} {'[2.64,2.64]':<14} "
          f"0.6824 0.6897 0.6648 0.6227 0.5341 0.4786 0.3900")

    # ---- LaTeX rows ----
    a2 = next(r for r in pooled_rows if r["variant"] == "A2")
    a3 = next(r for r in pooled_rows if r["variant"] == "A3")
    tex_path = os.path.join(OUT_DIR, "a23_table_rows.tex")
    with open(tex_path, "w") as f:
        f.write("% A2 + A3 LaTeX rows for Table 1 (5-fold CV held-out)\n")
        f.write("% A2: IPSsimRF re-tuned via 5-fold CV (RD-stability protocol)\n")
        f.write(
            "IPSsimRF (5-fold CV) & "
            + " & ".join(f"{a2[f'ndcg{k}']:.4f}" for k in CUTOFFS)
            + " \\\\\n"
        )
        f.write("% A3: no-audit α (ColBERT anchor + nDCG-grid 5-fold CV)\n")
        f.write(
            "IPSsimRF (no-audit $\\alpha$) & "
            + " & ".join(f"{a3[f'ndcg{k}']:.4f}" for k in CUTOFFS)
            + " \\\\\n"
        )
    print(f"\nSaved: {os.path.join(OUT_DIR, 'a23_per_fold.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'a23_pooled.csv')}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
