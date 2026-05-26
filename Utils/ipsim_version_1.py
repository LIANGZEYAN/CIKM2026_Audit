import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def remove_duplicates_on_text(df, qid_col="qid", text_col="text", rank_col="rank"):
    """
    Remove duplicate documents within each qid based on normalized text.
    Keep first occurrence and reassign consecutive ranks.
    """
    df = df.copy()
    df["_temp_index"] = range(len(df))
    df.sort_values(by=[qid_col, "_temp_index"], ascending=[True, True], inplace=True)
    
    duplicate_counts = {}
    list_of_subdfs = []
    
    for qid_val, sub_df in df.groupby(qid_col, group_keys=True):
        n_initial = len(sub_df)
        seen = set()
        keep_rows = []
        for _, row in sub_df.iterrows():
            txt = row[text_col]
            normalized_txt = str(txt).strip().lower()
            if normalized_txt not in seen:
                keep_rows.append(row)
                seen.add(normalized_txt)
        n_after = len(keep_rows)
        duplicate_counts[qid_val] = n_initial - n_after
        sub_df_dedup = pd.DataFrame(keep_rows)
        list_of_subdfs.append(sub_df_dedup)
    
    dedup_df = pd.concat(list_of_subdfs, ignore_index=True)
    dedup_df.sort_values(by="_temp_index", ascending=True, inplace=True)
    
    # Reassign consecutive ranks within each qid
    final_subdfs = []
    for qid_val, sub_df in dedup_df.groupby(qid_col, group_keys=False):
        sub_df[rank_col] = range(len(sub_df))
        final_subdfs.append(sub_df)
    final_df = pd.concat(final_subdfs, ignore_index=True)
    final_df.drop(columns=["_temp_index"], inplace=True)
    final_df.sort_values(by=[qid_col, rank_col], inplace=True, ascending=[True, True])
    final_df.reset_index(drop=True, inplace=True)
    
    total_removed = sum(duplicate_counts.values())
    print(f"Total duplicates removed: {total_removed}")
    print(f"Before: {len(df)}, After: {len(final_df)}")
    
    return final_df


def get_top_k_docs_by_score(df, qid_col="qid", docno_col="docno", score_col="score", top_k=5):
    """
    Get top-k documents with highest ColBERT scores for each qid.
    """
    top_docs_map = {}
    for qid_val, sub_df in df.groupby(qid_col):
        sorted_sub = sub_df.sort_values(by=score_col, ascending=False)
        top_docs = sorted_sub.head(top_k)[docno_col].tolist()
        top_docs_map[qid_val] = top_docs
    return top_docs_map


def add_semantic_similarity_colbert_top(
    df,
    top_docs_map,
    qid_col="qid",
    docno_col="docno",
    text_col="text",
    top_k=5,
    model_name="princeton-nlp/sup-simcse-bert-base-uncased",
    batch_size=32
):
    """
    Compute semantic similarity between each document and top-k ColBERT scored documents.
    Uses HuggingFace sentence embedding model.
    """
    df = df.copy()
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    def embed_texts_batch(texts, batch_size=32):
        """Generate embeddings for a list of texts in batches."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                             return_tensors="pt", max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embs)
        return embeddings
    
    # Generate embeddings for all texts
    all_texts = df[text_col].fillna("").astype(str).tolist()
    all_embeddings = embed_texts_batch(all_texts, batch_size=batch_size)
    df["embedding"] = all_embeddings
    
    def cosine_sim(a, b):
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    semantic_sim_dict = {}
    
    for qid_val, sub_df in df.groupby(qid_col):
        top_docnos = top_docs_map.get(qid_val, [])
        if not top_docnos:
            for idx in sub_df.index:
                semantic_sim_dict[idx] = 0.0
            continue
        
        # Get embeddings for top-k documents
        top_indices = sub_df[sub_df[docno_col].isin(top_docnos)].index.tolist()
        valid_embs = []
        for idx in top_indices:
            emb = df.at[idx, "embedding"]
            if emb is not None and isinstance(emb, np.ndarray):
                valid_embs.append(emb)
        
        if not valid_embs:
            for idx in sub_df.index:
                semantic_sim_dict[idx] = 0.0
            continue
        
        # Compute similarity for each document
        for idx, row in sub_df.iterrows():
            doc_emb = row["embedding"]
            if doc_emb is None or not isinstance(doc_emb, np.ndarray):
                semantic_sim_dict[idx] = 0.0
                continue
            
            # If document is in top-k, compute similarity with other top-k docs
            if row[docno_col] in top_docnos:
                ref_embs_list = []
                for ref_idx in top_indices:
                    ref_emb = df.at[ref_idx, "embedding"]
                    if ref_idx != idx and ref_emb is not None and isinstance(ref_emb, np.ndarray):
                        ref_embs_list.append(ref_emb)
                
                if not ref_embs_list:
                    avg_sim = 0.0
                else:
                    sims = [cosine_sim(doc_emb, ref_emb) for ref_emb in ref_embs_list]
                    avg_sim = np.mean(sims) if sims else 0.0
            else:
                # Compute similarity with all top-k docs
                sims = [cosine_sim(doc_emb, top_emb) for top_emb in valid_embs]
                avg_sim = np.mean(sims) if sims else 0.0
            
            semantic_sim_dict[idx] = avg_sim if not np.isnan(avg_sim) else 0.0
    
    df["semantic_sim"] = df.index.map(semantic_sim_dict).fillna(0.0)
    print("Semantic similarity computed and stored in 'semantic_sim' column.")
    return df


def compute_unbiased_score(
    df,
    alpha_coef=1.0,
    qid_col="qid",
    score_col="score",
    rank_col="rank",
    sim_col="semantic_sim"
):
    """
    Compute unbiased score using semantic similarity adjustment.
    
    Formula: unbiased_score = normalized_score · (1 + alpha_coef × semantic_sim)
    
    This is the SBR (Semantic-Based Reranking) approach without click data.
    """
    df = df.copy()
    
    def min_max_norm(series):
        """Min-max normalize within [0, 1]."""
        mn, mx = series.min(), series.max()
        if mx > mn:
            return (series - mn) / (mx - mn)
        else:
            return 0.5
    
    # Normalize ColBERT scores within each qid
    df["normalized_score"] = df.groupby(qid_col)[score_col].transform(min_max_norm)
    
    # Compute SBR score with semantic similarity adjustment
    df["sbr_score"] = df["normalized_score"] * (1 + alpha_coef * df[sim_col])
    
    # Generate new ranking based on SBR score
    df["sbr_rank"] = (
        df.groupby(qid_col)["sbr_score"]
          .rank(method="first", ascending=False)
          .astype(int)
    )
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Semantic-based reranking for ColBERT rankings without click data"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="colbert_rankings_default.csv",
        help="Path to ColBERT ranking CSV file (columns: qid, docno, score, text). Default: colbert_rankings_default.csv"
    )
    parser.add_argument(
        "top_k",
        nargs="?",
        type=int,
        default=5,
        help="Number of top documents to use for semantic similarity calculation. Default: 5"
    )
    parser.add_argument(
        "alpha_coef",
        nargs="?",
        type=float,
        default=1.0,
        help="Alpha coefficient for semantic similarity weight. Default: 1.0"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debiased_rankings_version1.csv",
        help="Output file path. Default: debiased_rankings_version1.csv"
    )
    
    args = parser.parse_args()
    
    print(f"Loading ColBERT rankings from: {args.input_file}")
    print(f"Parameters - Top-K: {args.top_k}, Alpha: {args.alpha_coef}")
    
    # Load ColBERT ranking results
    # Expected columns: qid, docno, score, text
    df = pd.read_csv(args.input_file)
    
    required_cols = ["qid", "docno", "score", "text"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
    
    print(f"Loaded {len(df)} documents for {df['qid'].nunique()} queries")
    
    # Step 1: Remove duplicates
    print("\nStep 1: Removing duplicate documents...")
    df_dedup = remove_duplicates_on_text(df, qid_col="qid", text_col="text", rank_col="rank")
    
    # Step 2: Get top-k documents by ColBERT score
    print(f"\nStep 2: Identifying top-{args.top_k} documents by ColBERT score...")
    top_docs_map = get_top_k_docs_by_score(df_dedup, top_k=args.top_k)
    
    # Step 3: Compute semantic similarity
    print("\nStep 3: Computing semantic similarity with top-k documents...")
    df_with_sim = add_semantic_similarity_colbert_top(
        df_dedup,
        top_docs_map,
        top_k=args.top_k,
        model_name="princeton-nlp/sup-simcse-bert-base-uncased",
        batch_size=32
    )
    
    # Step 4: Compute SBR score and re-rank
    print(f"\nStep 4: Computing SBR scores with alpha={args.alpha_coef}...")
    df_result = compute_unbiased_score(df_with_sim, alpha_coef=args.alpha_coef)
    
    # Step 5: Sort by qid and SBR score
    df_sorted = df_result.sort_values(by=["qid", "sbr_score"], ascending=[True, False])
    
    # Step 6: Save results
    print(f"\nSaving SBR rankings to: {args.output}")
    output_cols = ["qid", "docno", "score", "normalized_score", 
                   "semantic_sim", "sbr_score", "sbr_rank", "text"]
    df_sorted[output_cols].to_csv(args.output, index=False)
    
    print("\n=== Processing Complete ===")
    print(f"Output saved to: {args.output}")
    print(f"Total queries: {df_sorted['qid'].nunique()}")
    print(f"Total documents: {len(df_sorted)}")


if __name__ == "__main__":
    main()
