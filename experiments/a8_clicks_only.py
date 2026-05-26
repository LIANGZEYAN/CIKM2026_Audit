"""
A8 — Click-only vs click+selection ablation (CIKM 2026 resubmission).

Reproduces the IPSsimRF pipeline from `utils/case_study_rankings/ipssim_version_2.ipynb`
twice:

  V1 (sanity, = original IPSsimRF):
      audit anchor = top-3 PASSAGE_SELECTION docs per qid (from study1 SQL)
      expected nDCG@10 = 0.6227 (matches Table 1)

  V2 (A8 clicks-only):
      audit anchor = top-3 OPEN_DOC docs per qid (from study1 SQL)

For each variant: grid-search alpha in [0, 5] step 0.1 to maximise nDCG@10 over
the 53 study queries, then report nDCG@{1,3,5,10,30,50,100} at the best alpha.

Pipeline (cells 1-12 of ipssim_version_2.ipynb):
  1. Load final_df_colbert_historical.csv, deduplicate by text per qid
  2. Simulate clicks with affine PBM (seed=505, total_clicks=1,000,000)
  3. Compute per-qid CTR + biased rank
  4. Build anchor map = top-3 docs by audit-event count (variant-specific)
  5. SimCSE cosine similarity from every doc to its qid's anchors (avg, excl self)
  6. Per qid, min-max normalise CTR
  7. Score = (normalized_ctr - beta + alpha * semantic_sim) / max(1.0, 100/sqrt(total_clicks))
  8. Evaluate vs TREC DL 2021 qrels via pt.Evaluate

Outputs:
  - a8_alpha_grid.csv         (variant, alpha, ndcg10)
  - a8_summary.csv            (variant, best_alpha, ndcg@1..100)
  - a8_table_row.tex          (one LaTeX row for Table 1: "IPSsimRF (clicks only)")
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import pyterrier as pt
if not pt.java.started():
    pt.java.init()
from ir_measures import nDCG, calc_aggregate

# ----- Config -----
HIST_PATH = "/mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv"
SQL_PATH = "/mnt/primary/Trec-llm/utils/case_study_rankings/railway_database_backup_v1.sql"
QRELS_PATH = "/mnt/primary/Trec-llm/dataset/msmarco-passage-v2/trec-dl-2021/qrels"
OUT_DIR = "/mnt/primary/cikm 2026/experiments"
SIMCSE_MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"
HF_CACHE = "/mnt/primary/huggingface_cache"

TOP_K_ANCHOR = 3
SIM_SEED = 505
TOTAL_CLICKS = 1_000_000
ALPHA_GRID = [round(x, 2) for x in np.arange(0.0, 5.01, 0.1)]
PARAM_BETA = 0.0
CUTOFFS = [1, 3, 5, 10, 30, 50, 100]

os.environ["HF_HOME"] = HF_CACHE


# ===== SQL parsing (B6 helper, reused) =====
LOGS_BLOCK_RE = re.compile(r"INSERT INTO `logs` VALUES (.*?);", re.DOTALL)
LOGS_ROW_RE = re.compile(
    r"\(\s*"
    r"(\d+),'([^']*)',(\d+),'([^']*)','([^']*)',"
    r"(-?\d+),(-?\d+),(\d+),(\d+),'([^']*)'\)"
)


def parse_logs(sql_path):
    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        sql = f.read()
    rows = []
    for block in LOGS_BLOCK_RE.findall(sql):
        for m in LOGS_ROW_RE.finditer(block):
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=[
        "id", "user_id", "qid", "docno", "event_type",
        "start_idx", "end_idx", "duration", "pass_flag", "timestamp",
    ])
    df["qid"] = df["qid"].astype(int)
    df["pass_flag"] = df["pass_flag"].astype(int)
    return df


# ===== Pipeline cells =====
def remove_duplicates_on_text(df, qid_col="qid", text_col="text", rank_col="rank"):
    """Cell 1: dedupe by lowercased+stripped text within each qid; reassign rank."""
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
        sub[rank_col] = range(len(sub))
        finals.append(sub)
    final = pd.concat(finals, ignore_index=True)
    final.sort_values([qid_col, rank_col], inplace=True)
    final.reset_index(drop=True, inplace=True)
    return final


def simulate_affine_clicks(df, total_clicks_target=TOTAL_CLICKS, seed=SIM_SEED):
    """Cell 5: simulate clicks per affine PBM; deterministic with seed."""
    np.random.seed(seed)
    df = df.copy()
    # per-qid min-max normalise score
    def _mm(s):
        a, b = s.min(), s.max()
        return (s - a) / (b - a) if b > a else pd.Series([0.5] * len(s), index=s.index)
    df["normalized_score"] = df.groupby("qid")["score"].transform(_mm)
    df["alpha"] = df["rank"].apply(lambda k: 1.0 / (1.0 + np.log1p(k + 1)))
    df["beta"] = df["rank"].apply(lambda k: 0.1 / (1.0 + (k + 1)))
    df["click_probability"] = df["alpha"] * df["normalized_score"] + df["beta"]
    # Distribute clicks across qids proportional to total click_probability
    total_p = df["click_probability"].sum()
    qid_clicks = {}
    for qid, sub in df.groupby("qid"):
        share = sub["click_probability"].sum() / total_p
        qid_clicks[qid] = max(1, int(round(total_clicks_target * share)))
    # Multinomial within each qid
    counts = []
    for qid, sub in df.groupby("qid", sort=False):
        n = qid_clicks[qid]
        probs = sub["click_probability"].values
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        ks = np.random.multinomial(n, probs)
        counts.extend(ks.tolist())
    df["click_count"] = counts
    return df


def add_normalized_ctr(df, qid_col="qid", ctr_col="ctr"):
    """Cell 9 helper: per-qid min-max normalisation of CTR."""
    df = df.copy()
    def _mm(s):
        a, b = s.min(), s.max()
        return (s - a) / (b - a) if b > a else pd.Series([0.5] * len(s), index=s.index)
    df["normalized_ctr"] = df.groupby(qid_col)[ctr_col].transform(_mm)
    return df


# ===== Anchor builders =====
def build_anchors_from_logs(df_logs, event_type, top_k=TOP_K_ANCHOR):
    """Per qid, get top-K docnos with the most events of `event_type`."""
    e = df_logs[df_logs["event_type"] == event_type]
    counts = e.groupby(["qid", "docno"]).size().reset_index(name="n")
    out = {}
    for qid, sub in counts.groupby("qid"):
        top = sub.nlargest(top_k, "n")["docno"].astype(str).tolist()
        out[int(qid)] = top
    return out


# ===== SimCSE =====
def load_simcse():
    print(f"  Loading SimCSE: {SIMCSE_MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
    model = AutoModel.from_pretrained(SIMCSE_MODEL)
    model.eval()
    return tok, model


def embed_texts(tok, model, texts, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768))


def avg_cosine_to_anchors(target_emb, anchor_embs, target_docno, anchor_docnos):
    """Average cosine sim of target to anchors, excluding self."""
    sims = []
    for a_emb, a_doc in zip(anchor_embs, anchor_docnos):
        if a_doc == target_docno:
            continue
        na = np.linalg.norm(target_emb)
        nb = np.linalg.norm(a_emb)
        if na > 0 and nb > 0:
            sims.append(float(np.dot(target_emb, a_emb) / (na * nb)))
    return float(np.mean(sims)) if sims else 0.0


# ===== nDCG eval =====
def load_qrels():
    qrels = pd.read_csv(
        QRELS_PATH, sep=r"\s+", header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )
    return qrels[["query_id", "doc_id", "relevance"]]


def eval_ndcg(df, qrels, cutoffs=CUTOFFS):
    """df has qid, docno, score; returns dict {k: ndcg@k_aggregated}."""
    run = df[["qid", "docno", "score"]].rename(
        columns={"qid": "query_id", "docno": "doc_id"}
    )
    run["query_id"] = run["query_id"].astype(str)
    run["doc_id"] = run["doc_id"].astype(str)
    measures = [nDCG @ k for k in cutoffs]
    res = calc_aggregate(measures, qrels, run)
    out = {}
    for m, v in res.items():
        m_str = str(m)
        k = int(m_str.split("@")[1])
        out[k] = float(v)
    return out


# ===== Main =====
def main():
    print("Step 1/8: Load + dedupe historical pool ...", flush=True)
    hist_raw = pd.read_csv(HIST_PATH, lineterminator="\n")
    hist_raw["qid"] = hist_raw["qid"].astype(int)
    hist_raw["docno"] = hist_raw["docno"].astype(str)
    hist_raw["score"] = hist_raw["score"].astype(float)
    hist = remove_duplicates_on_text(hist_raw, qid_col="qid", text_col="text", rank_col="rank")
    print(f"  After dedup: {len(hist)} rows, {hist['qid'].nunique()} qids, "
          f"docs/qid mean={hist.groupby('qid').size().mean():.1f}")

    print("Step 2/8: Simulate clicks (PBM, seed=505, total=1e6) ...", flush=True)
    sim = simulate_affine_clicks(hist[["qid", "query", "docno", "text", "score", "rank"]].copy())
    sim["ctr"] = sim.groupby("qid")["click_count"].transform(lambda x: x / x.sum())
    sim["biased_rank"] = sim.groupby("qid")["ctr"].rank(method="first", ascending=False).astype(int)
    sim = add_normalized_ctr(sim, qid_col="qid", ctr_col="ctr")
    print(f"  Simulated. Total clicks = {int(sim['click_count'].sum()):,}")

    print("Step 3/8: Parse Study 1 SQL ...", flush=True)
    logs = parse_logs(SQL_PATH)
    print(f"  Logs rows: {len(logs)}, distinct event_types: "
          f"{logs['event_type'].value_counts().head(5).to_dict()}")

    anchors_v1 = build_anchors_from_logs(logs, "PASSAGE_SELECTION", TOP_K_ANCHOR)
    anchors_v2 = build_anchors_from_logs(logs, "OPEN_DOC", TOP_K_ANCHOR)
    print(f"  V1 selection-anchor map: {len(anchors_v1)} qids")
    print(f"  V2 open-doc-anchor map:   {len(anchors_v2)} qids")
    overlap = sum(1 for q in anchors_v1
                  if anchors_v1[q] == anchors_v2[q])
    print(f"  qids where V1 anchor == V2 anchor: {overlap}/{len(anchors_v1)}")

    print("Step 4/8: SimCSE embed all docs in pool ...", flush=True)
    tok, model = load_simcse()
    qid_doc_pairs = list(zip(sim["qid"].tolist(), sim["docno"].tolist()))
    texts = sim["text"].fillna("").astype(str).tolist()
    embs = embed_texts(tok, model, texts, batch_size=16)
    emb_lookup = {(qid_doc_pairs[i][0], qid_doc_pairs[i][1]): embs[i] for i in range(len(embs))}
    print(f"  Embedded {len(emb_lookup)} docs")

    # ----- Compute sim_NEW for each variant -----
    def compute_sim_per_doc(df, anchors_map):
        sims = np.empty(len(df))
        for i, (qid, docno) in enumerate(zip(df["qid"].values, df["docno"].values)):
            target = emb_lookup.get((int(qid), str(docno)))
            anchors = anchors_map.get(int(qid), [])
            anchor_embs = [emb_lookup.get((int(qid), a)) for a in anchors]
            anchor_embs = [(e, a) for e, a in zip(anchor_embs, anchors) if e is not None]
            if target is None or not anchor_embs:
                sims[i] = 0.0
                continue
            anchor_es = [e for e, _ in anchor_embs]
            anchor_docs = [a for _, a in anchor_embs]
            sims[i] = avg_cosine_to_anchors(target, anchor_es, str(docno), anchor_docs)
        return sims

    print("Step 5/8: Compute sim_clicks (V1 selection) ...", flush=True)
    sim["semantic_sim_v1"] = compute_sim_per_doc(sim, anchors_v1)
    print("Step 6/8: Compute sim_clicks (V2 open-doc) ...", flush=True)
    sim["semantic_sim_v2"] = compute_sim_per_doc(sim, anchors_v2)

    qrels = load_qrels()
    lower_bound_alpha = 100.0 / np.sqrt(TOTAL_CLICKS)  # 0.1
    param_alpha_clipped = max(1.0, lower_bound_alpha)  # = 1.0

    def score_and_eval(sim_col, alpha):
        # Build a fresh run frame; do NOT inherit the original ColBERT `score` column.
        run = sim[["qid", "docno", "normalized_ctr", sim_col]].copy()
        run["score"] = (
            (run["normalized_ctr"] - PARAM_BETA + alpha * run[sim_col]) / param_alpha_clipped
        )
        run = run[["qid", "docno", "score"]]
        return eval_ndcg(run, qrels, CUTOFFS)

    print("Step 7/8: Grid α∈[0,5] step 0.1 for both variants ...", flush=True)
    grid_rows = []
    for variant_label, sim_col in [("V1_PASSAGE_SELECTION", "semantic_sim_v1"),
                                    ("V2_OPEN_DOC_only", "semantic_sim_v2")]:
        for a in ALPHA_GRID:
            res = score_and_eval(sim_col, a)
            grid_rows.append({"variant": variant_label, "alpha": a,
                              "ndcg10": res[10], "ndcg1": res[1], "ndcg3": res[3],
                              "ndcg5": res[5], "ndcg30": res[30], "ndcg50": res[50],
                              "ndcg100": res[100]})
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(OUT_DIR, "a8_alpha_grid.csv"), index=False)

    print("Step 8/8: Pick best α per variant; eval all cutoffs ...", flush=True)
    summary_rows = []
    for variant in ["V1_PASSAGE_SELECTION", "V2_OPEN_DOC_only"]:
        sub = grid_df[grid_df["variant"] == variant]
        best_idx = sub["ndcg10"].idxmax()
        best = sub.loc[best_idx]
        rec = {"variant": variant, "best_alpha": float(best["alpha"])}
        for k in CUTOFFS:
            rec[f"ndcg{k}"] = float(best[f"ndcg{k}"])
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "a8_summary.csv"), index=False)

    # ----- Pretty print -----
    print("\n" + "=" * 90)
    print("A8 RESULTS — best-α IPSsimRF, two variants")
    print("=" * 90)
    print(f"{'Variant':<25} {'best_α':>8} {'@1':>8} {'@3':>8} {'@5':>8} "
          f"{'@10':>8} {'@30':>8} {'@50':>8} {'@100':>8}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['variant']:<25} {r['best_alpha']:>8.2f} "
              f"{r['ndcg1']:>8.4f} {r['ndcg3']:>8.4f} {r['ndcg5']:>8.4f} "
              f"{r['ndcg10']:>8.4f} {r['ndcg30']:>8.4f} {r['ndcg50']:>8.4f} "
              f"{r['ndcg100']:>8.4f}")

    print("\nReference (paper Table 1, IPSsimRF α=2.6429):")
    print(f"{'IPSsimRF (paper)':<25} {2.6429:>8.4f} 0.6824   0.6897   0.6648   "
          f"0.6227   0.5341   0.4786   0.3900")

    # ----- LaTeX row -----
    v2 = [r for r in summary_rows if r["variant"] == "V2_OPEN_DOC_only"][0]
    latex_path = os.path.join(OUT_DIR, "a8_table_row.tex")
    with open(latex_path, "w") as f:
        f.write("% A8 row to slot into Table 1 (tab:relevance_feedback_results)\n")
        f.write(
            "IPSsimRF (clicks only) & "
            f"{v2['ndcg1']:.4f} & {v2['ndcg3']:.4f} & {v2['ndcg5']:.4f} & "
            f"{v2['ndcg10']:.4f} & {v2['ndcg30']:.4f} & {v2['ndcg50']:.4f} & "
            f"{v2['ndcg100']:.4f} \\\\\n"
        )
    print(f"\nSaved LaTeX row: {latex_path}")


if __name__ == "__main__":
    main()
