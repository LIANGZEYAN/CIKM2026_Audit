"""
C3 — ASR-Max feature analysis (CIKM 2026 resubmission).

Identifies the 13 ASR-Max documents from Study 1 (SBR-surfaced docs whose
selection count exceeded all 4 top-4 ColBERT docs in the same query) and
computes 5 features for 4 groups for qualitative comparison:

Groups:
  A: 13 ASR-Max docs                  (the strongest "audit success" cases)
  B: All 212 SBR-surfaced docs        (broader candidate pool, includes A)
  C: All 212 ColBERT top-4 docs       (top-ranked baseline)
  D: Broader historical pool          (~5300 docs from the deeper ColBERT ranking)

Features:
  1. Passage length            (word count)
  2. Query–passage Jaccard     (lowercased word-token Jaccard)
  3. ColBERT score percentile  (rank within qid's historical pool, 100% = top)
  4. SimCSE similarity to top-4 ColBERT docs (avg cosine; not computed for group D)
  5. Named-entity density      (entities per 100 words via spaCy en_core_web_sm)

With n=13 in group A, treat the comparison qualitatively rather than as
inferential statistics. Output a small features-by-group means+SD table.

Inputs:
  - chiir_plot_pdf/query_doc_clicks_matrix_with_ids_study1.csv
  - case_study_rankings/final_df_colbert_historical.csv
  - SimCSE: princeton-nlp/sup-simcse-bert-base-uncased
  - spaCy en_core_web_sm

Outputs:
  - c3_asr_max_docs.csv          (the 13 ASR-Max docs with their selection counts)
  - c3_per_doc_features.csv      (per-doc features for groups A, B, C; not D — too big)
  - c3_features_by_group.csv     (group means and SD)
  - c3_table.tex                 (LaTeX-ready table for paper Section 5)
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import spacy

# ----- Config -----
MATRIX_PATH = "/mnt/primary/Trec-llm/colbert_ipssim/chiir_plot_pdf/query_doc_clicks_matrix_with_ids_study1.csv"
HIST_PATH = "/mnt/primary/Trec-llm/utils/case_study_rankings/final_df_colbert_historical.csv"
OUT_DIR = "/mnt/primary/cikm 2026/experiments"
SIMCSE_MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"
HF_CACHE = "/mnt/primary/huggingface_cache"
TOP_K_ANCHOR = 4   # ColBERT top-K used as similarity anchor (matches paper's D_top)
NER_PER = 100      # entities per 100 words

os.makedirs(OUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE


# ===== Step 1: Identify ASR-Max docs =====
def find_asr_max():
    """Return list of dicts: {qid, docno, sbr_pos, selections, colbert_max} for the 13 ASR-Max."""
    m = pd.read_csv(MATRIX_PATH)
    asr_max = []
    for _, row in m.iterrows():
        qid = int(row["qid"])
        cb = [int(row[f"colbert_{k}_clicks"]) for k in [1, 2, 3, 4]]
        cb_max = max(cb)
        for k in [1, 2, 3, 4]:
            sbr_id = row[f"ipssim_{k}_id"]
            sbr_n = int(row[f"ipssim_{k}_clicks"])
            if sbr_n > cb_max:
                asr_max.append({
                    "qid": qid, "docno": sbr_id, "sbr_pos": k,
                    "selections": sbr_n, "colbert_max": cb_max
                })
    return asr_max


def build_groups(asr_max_recs):
    """Build (qid, docno) tuples for each group + texts/queries."""
    m = pd.read_csv(MATRIX_PATH)
    group_A = [(r["qid"], r["docno"]) for r in asr_max_recs]   # 13
    group_B = []  # 212 SBR
    group_C = []  # 212 ColBERT top-4
    for _, row in m.iterrows():
        qid = int(row["qid"])
        for k in [1, 2, 3, 4]:
            group_B.append((qid, row[f"ipssim_{k}_id"]))
            group_C.append((qid, row[f"colbert_{k}_id"]))
    return group_A, group_B, group_C


# ===== Step 2: Load text + ColBERT score per (qid, docno) =====
def load_doc_lookup():
    """Build dict: (qid, docno) -> (query, text, colbert_score, rank_in_qid_pool, pool_size)."""
    df = pd.read_csv(HIST_PATH, lineterminator="\n")
    df["qid"] = df["qid"].astype(int)
    df["docno"] = df["docno"].astype(str)
    df["score"] = df["score"].astype(float)
    df = df.sort_values(["qid", "score"], ascending=[True, False])
    df = df.drop_duplicates(["qid", "docno"], keep="first")

    # Compute pool size + rank within qid
    df["pool_size"] = df.groupby("qid")["docno"].transform("count")
    df["rank_in_pool"] = df.groupby("qid")["score"].rank(method="first", ascending=False).astype(int)
    df["colbert_percentile"] = 100 * (1 - (df["rank_in_pool"] - 1) / (df["pool_size"] - 1).clip(lower=1))

    lookup = {}
    for _, r in df.iterrows():
        lookup[(int(r["qid"]), str(r["docno"]))] = {
            "query": r["query"],
            "text": r["text"] if isinstance(r["text"], str) else "",
            "colbert_score": float(r["score"]),
            "rank_in_pool": int(r["rank_in_pool"]),
            "pool_size": int(r["pool_size"]),
            "colbert_percentile": float(r["colbert_percentile"]),
        }
    return lookup, df


# ===== Step 3: Feature computers =====
TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


def passage_length(text):
    return len(TOKEN_RE.findall(text.lower())) if text else 0


def query_jaccard(query, text):
    if not query or not text:
        return 0.0
    qtok = set(TOKEN_RE.findall(query.lower()))
    ptok = set(TOKEN_RE.findall(text.lower()))
    if not qtok and not ptok:
        return 0.0
    union = qtok | ptok
    inter = qtok & ptok
    return len(inter) / len(union) if union else 0.0


def ner_count_batch(nlp, texts, batch_size=64):
    """Batch-NER via spaCy pipe(). Returns list of #entities per text."""
    counts = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        counts.append(len(doc.ents))
    return counts


# ===== Step 4: SimCSE similarity =====
def load_simcse():
    print(f"  Loading SimCSE: {SIMCSE_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
    model = AutoModel.from_pretrained(SIMCSE_MODEL)
    model.eval()
    return tok, model


def embed_texts(tok, model, texts, batch_size=16):
    """Return ndarray (N, D)."""
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768))


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ===== Main =====
def main():
    print("Step 1: Identify ASR-Max docs ...")
    asr_max_recs = find_asr_max()
    print(f"  Found {len(asr_max_recs)} ASR-Max docs (paper says 13)")
    df_asr = pd.DataFrame(asr_max_recs)
    df_asr.to_csv(os.path.join(OUT_DIR, "c3_asr_max_docs.csv"), index=False)

    group_A, group_B, group_C = build_groups(asr_max_recs)
    print(f"  Group A (ASR-Max): {len(group_A)} docs")
    print(f"  Group B (all SBR): {len(group_B)} cells")
    print(f"  Group C (ColBERT top-4): {len(group_C)} cells")

    print("\nStep 2: Load doc texts + ColBERT scores ...")
    lookup, hist_df = load_doc_lookup()
    print(f"  Historical pool: {len(hist_df)} unique (qid,docno) over {hist_df['qid'].nunique()} qids")

    # Sanity: how many docs in groups A/B/C exist in historical pool?
    for name, group in [("A", group_A), ("B", group_B), ("C", group_C)]:
        missing = sum(1 for k in group if k not in lookup)
        print(f"  Group {name}: {len(group) - missing}/{len(group)} found in historical lookup ({missing} missing)")

    # Build ColBERT top-K anchor docnos per qid (from historical pool — top scores)
    print(f"\nStep 3: Build ColBERT top-{TOP_K_ANCHOR} anchor per qid ...")
    anchors_by_qid = {}
    for qid, sub in hist_df.groupby("qid"):
        top_docs = sub.nlargest(TOP_K_ANCHOR, "score")["docno"].astype(str).tolist()
        anchors_by_qid[int(qid)] = top_docs

    # Collect unique docs that need SimCSE embedding (groups A+B+C + anchors)
    unique_docs = set()
    for group in [group_A, group_B, group_C]:
        for k in group:
            unique_docs.add(k)
    for qid, anchors in anchors_by_qid.items():
        for d in anchors:
            unique_docs.add((qid, d))
    print(f"  Unique (qid,docno) needing embedding: {len(unique_docs)}")

    unique_keys = [k for k in unique_docs if k in lookup]
    texts = [lookup[k]["text"] for k in unique_keys]
    missing = len(unique_docs) - len(unique_keys)
    if missing:
        print(f"  WARNING: {missing} docs missing from historical pool — will skip SimCSE for them")

    print("\nStep 4: SimCSE embeddings ...")
    tok, model = load_simcse()
    embs = embed_texts(tok, model, texts, batch_size=16)
    emb_lookup = {k: e for k, e in zip(unique_keys, embs)}
    print(f"  Embedded {len(emb_lookup)} docs")

    print("\nStep 5: Load spaCy NER model ...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    print("\nStep 6: Compute per-doc features for groups A/B/C ...", flush=True)
    abc_rows = []
    for group_name, group in [("A_ASR_Max", group_A), ("B_All_SBR", group_B), ("C_ColBERT_top4", group_C)]:
        for (qid, docno) in group:
            info = lookup.get((qid, docno))
            if info is None:
                continue
            text = info["text"] or ""
            query = info["query"] or ""
            length = passage_length(text)
            jacc = query_jaccard(query, text)
            cb_pct = info["colbert_percentile"]

            # SimCSE: avg cosine to ColBERT top-K anchors of same qid (exclude self)
            anchors = anchors_by_qid.get(int(qid), [])
            target_emb = emb_lookup.get((qid, docno))
            sims = []
            if target_emb is not None:
                for a_doc in anchors:
                    if a_doc == docno:
                        continue
                    a_emb = emb_lookup.get((qid, a_doc))
                    if a_emb is None:
                        continue
                    sims.append(cosine(target_emb, a_emb))
            simcse_avg = float(np.mean(sims)) if sims else np.nan

            abc_rows.append({
                "group": group_name, "qid": qid, "docno": docno,
                "text": text, "query_text": query,
                "passage_length_words": length, "query_jaccard": jacc,
                "colbert_percentile": cb_pct, "simcse_to_top4": simcse_avg,
            })

    print(f"  Built {len(abc_rows)} rows for groups A/B/C", flush=True)
    print("  NER batch on A/B/C (~437 docs) ...", flush=True)
    abc_ner = ner_count_batch(nlp, [r["text"] for r in abc_rows], batch_size=64)
    for r, n_ents in zip(abc_rows, abc_ner):
        r["ner_density_per100w"] = NER_PER * n_ents / r["passage_length_words"] if r["passage_length_words"] > 0 else 0.0

    feat_df = pd.DataFrame(abc_rows).drop(columns=["text", "query_text"])
    feat_df.to_csv(os.path.join(OUT_DIR, "c3_per_doc_features.csv"), index=False)

    # Group D: features for full historical pool (no SimCSE — too expensive, but batched NER)
    print("\nStep 7: Compute group D (broader pool, ~5300 docs) features (no SimCSE) ...", flush=True)
    d_rows = hist_df[["qid", "docno", "text", "query", "colbert_percentile"]].copy()
    d_rows["text"] = d_rows["text"].fillna("").astype(str)
    d_rows["query"] = d_rows["query"].fillna("").astype(str)
    d_rows["passage_length_words"] = d_rows["text"].apply(passage_length)
    d_rows["query_jaccard"] = d_rows.apply(lambda r: query_jaccard(r["query"], r["text"]), axis=1)
    print(f"  NER batch on Group D ({len(d_rows)} docs) ...", flush=True)
    d_ner = ner_count_batch(nlp, d_rows["text"].tolist(), batch_size=64)
    d_rows["ner_density_per100w"] = [
        NER_PER * n_ents / lw if lw > 0 else 0.0
        for n_ents, lw in zip(d_ner, d_rows["passage_length_words"])
    ]
    d_rows["simcse_to_top4"] = np.nan
    d_rows["group"] = "D_Broader_Pool"
    d_rows["qid"] = d_rows["qid"].astype(int)
    d_df = d_rows[["group", "qid", "docno", "passage_length_words", "query_jaccard",
                   "colbert_percentile", "simcse_to_top4", "ner_density_per100w"]]

    # Combine feature CSVs
    full_feat = pd.concat([feat_df, d_df], ignore_index=True)
    full_feat.to_csv(os.path.join(OUT_DIR, "c3_per_doc_features_full.csv"), index=False)

    # ===== Aggregation =====
    print("\nStep 8: Aggregate to group means/SD ...")
    feature_cols = ["passage_length_words", "query_jaccard", "colbert_percentile",
                    "simcse_to_top4", "ner_density_per100w"]
    summary_rows = []
    for group_name, group_df in [
        ("A: ASR-Max (n=13)", feat_df[feat_df["group"] == "A_ASR_Max"]),
        ("B: All SBR (n=212)", feat_df[feat_df["group"] == "B_All_SBR"]),
        ("C: ColBERT top-4 (n=212)", feat_df[feat_df["group"] == "C_ColBERT_top4"]),
        ("D: Broader pool (n=%d)" % len(d_df), d_df),
    ]:
        rec = {"group": group_name, "n": len(group_df)}
        for c in feature_cols:
            vals = group_df[c].dropna()
            if len(vals) == 0:
                rec[f"{c}_mean"] = np.nan
                rec[f"{c}_sd"] = np.nan
            else:
                rec[f"{c}_mean"] = float(vals.mean())
                rec[f"{c}_sd"] = float(vals.std())
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "c3_features_by_group.csv"), index=False)

    # Pretty print
    print("\n" + "=" * 80)
    print("FEATURES BY GROUP (mean ± SD)")
    print("=" * 80)
    short_labels = {
        "passage_length_words": "Length (words)",
        "query_jaccard": "Q-Jaccard",
        "colbert_percentile": "ColBERT %ile",
        "simcse_to_top4": "SimCSE→top-4",
        "ner_density_per100w": "NER /100w",
    }
    cols = ["group", "n"] + feature_cols
    header = ["Group", "n"] + [short_labels[c] for c in feature_cols]
    rows_pretty = []
    for _, r in summary.iterrows():
        row = [r["group"], int(r["n"])]
        for c in feature_cols:
            m, sd = r[f"{c}_mean"], r[f"{c}_sd"]
            if np.isnan(m):
                row.append("—")
            else:
                row.append(f"{m:.2f}±{sd:.2f}")
        rows_pretty.append(row)

    widths = [max(len(str(r[i])) for r in [header] + rows_pretty) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows_pretty:
        print(fmt.format(*[str(x) for x in r]))

    # ===== LaTeX Table =====
    latex_path = os.path.join(OUT_DIR, "c3_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by c3_asr_max_features.py\n")
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \caption{Feature profile of the 13 ASR-Max audit-success documents (Group A) compared with the broader audit-candidate pool (B, all SBR-surfaced cells), the top-ranked ColBERT documents (C), and the wider ColBERT pool (D, $\sim$100 docs/query). Means $\pm$ SD; with $n{=}13$ in Group A, comparisons are descriptive rather than inferential. SimCSE similarity is to ColBERT top-4 of the same query (excluding self).}" + "\n")
        f.write(r"  \label{tab:c3_asr_max_features}" + "\n")
        f.write(r"  \small" + "\n")
        f.write(r"  \begin{tabular}{lrrrrrr}" + "\n")
        f.write(r"    \toprule" + "\n")
        f.write(r"    Group & $n$ & Length & Q-Jacc. & ColBERT \%ile & SimCSE & NER/100w \\" + "\n")
        f.write(r"    \midrule" + "\n")
        for r in rows_pretty:
            cells = [r[0], str(r[1])] + r[2:]
            f.write("    " + " & ".join(cells).replace("±", r"$\pm$") + r" \\" + "\n")
        f.write(r"    \bottomrule" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\nSaved LaTeX: {latex_path}")
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
