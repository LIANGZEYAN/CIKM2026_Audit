import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine


def strategic_document_selection(colbert_ranking_df, sbr_ranking_df, qid, qrels_df=None, top_k=4, 
                                model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Strategic document selection: picks top documents from both rankings and adds a dissimilar negative example.
    
    Selection strategy:
    1. Select top-k documents from ColBERT ranking
    2. Select top-k documents from SBR ranking (excluding duplicates)
    3. Select 1 easy negative document (lowest semantic similarity, label=0)
    
    Total: 2*top_k + 1 documents (default: 4 + 4 + 1 = 9 documents)
    """
    # Filter data for the specified qid
    colbert_df = colbert_ranking_df[colbert_ranking_df['qid'] == qid].copy()
    sbr_df = sbr_ranking_df[sbr_ranking_df['qid'] == qid].copy()
    
    if colbert_df.empty or sbr_df.empty:
        raise ValueError(f"qid={qid} is missing in one/both rankings")
    
    query_text = colbert_df['query'].iloc[0]
    
    # Sort by respective rank columns
    colbert_sorted = colbert_df.sort_values('colbert_rank').reset_index(drop=True)
    sbr_sorted = sbr_df.sort_values('sbr_rank').reset_index(drop=True)
    
    selected_docs = []
    selected_docnos = set()
    
    # Step 1: Select top-k documents from ColBERT ranking
    for i in range(min(top_k, len(colbert_sorted))):
        doc = colbert_sorted.iloc[i]
        
        # Find SBR rank if exists
        sbr_rank = np.nan
        matching_sbr = sbr_df[sbr_df['docno'] == doc['docno']]
        if not matching_sbr.empty:
            sbr_rank = matching_sbr.iloc[0]['sbr_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'colbert_rank': doc['colbert_rank'],
            'sbr_rank': sbr_rank,
            'source': "top from ColBERT",
            'from': "ColBERT",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
    
    # Step 2: Select top-k documents from SBR ranking (skip duplicates)
    sbr_idx = 0
    selected_from_sbr = 0
    
    while selected_from_sbr < top_k and sbr_idx < len(sbr_sorted):
        doc = sbr_sorted.iloc[sbr_idx]
        sbr_idx += 1
        
        if doc['docno'] in selected_docnos:
            continue
        
        # Find ColBERT rank if exists
        colbert_rank = np.nan
        matching_colbert = colbert_df[colbert_df['docno'] == doc['docno']]
        if not matching_colbert.empty:
            colbert_rank = matching_colbert.iloc[0]['colbert_rank']
            
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'colbert_rank': colbert_rank,
            'sbr_rank': doc['sbr_rank'],
            'source': "top from SBR",
            'from': "SBR",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
        selected_from_sbr += 1
    
    # If no remaining documents, return what we have
    remaining_docs = sbr_sorted[~sbr_sorted['docno'].isin(selected_docnos)].copy()
    if remaining_docs.empty:
        return pd.DataFrame(selected_docs)
    
    # Step 3: Compute semantic similarity for easy negative selection
    temp_df = pd.DataFrame(selected_docs)
    all_docs_df = pd.concat([remaining_docs, temp_df]).reset_index(drop=True)
    
    try:
        docs_with_sim = add_semantic_similarity_hf(
            all_docs_df,
            qid_col="qid",
            text_col="text",
            rank_col="sbr_rank" if "sbr_rank" in all_docs_df.columns else "sbr_rank",
            top_k=min(len(temp_df), 5),
            model_name=model_name,
            batch_size=32
        )
    except Exception as e:
        print(f"Error computing semantic similarity: {str(e)}")
        return pd.DataFrame(selected_docs)
    
    # Step 4: Find easy negative from remaining documents
    neg_candidates_df = docs_with_sim[~docs_with_sim['docno'].isin(selected_docnos)].copy()
    
    # Filter for label=0 documents if qrels provided
    if qrels_df is not None and not qrels_df.empty and 'qid' in qrels_df.columns:
        query_qrels = qrels_df[qrels_df['qid'] == qid]
        
        if not query_qrels.empty:
            neg_candidates_df = neg_candidates_df.merge(
                query_qrels[['docno', 'label']], 
                on='docno',
                how='left'
            )
            neg_candidates_df['label'] = neg_candidates_df['label'].fillna(0)
            neg_candidates_rel0 = neg_candidates_df[neg_candidates_df['label'] == 0]
            
            if not neg_candidates_rel0.empty:
                neg_candidates_df = neg_candidates_rel0
    
    # Select document with lowest semantic similarity
    if not neg_candidates_df.empty:
        neg_candidates_df = neg_candidates_df.sort_values('semantic_sim')
        neg_doc = neg_candidates_df.iloc[0]
        
        # Find ColBERT rank
        colbert_rank = np.nan
        matching_colbert = colbert_df[colbert_df['docno'] == neg_doc['docno']]
        if not matching_colbert.empty:
            colbert_rank = matching_colbert.iloc[0]['colbert_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': neg_doc['docno'],
            'text': neg_doc['text'],
            'colbert_rank': colbert_rank,
            'sbr_rank': neg_doc['sbr_rank'] if 'sbr_rank' in neg_doc else neg_doc['sbr_rank'],
            'semantic_sim': neg_doc['semantic_sim'],
            'source': "easy negative",
            'from': "SBR",
            'selected_in_turn': len(selected_docs) + 1,
            'label': neg_doc['label'] if 'label' in neg_doc else np.nan
        })
    
    return pd.DataFrame(selected_docs)


def add_semantic_similarity_hf(
    df,
    qid_col="qid",
    text_col="text",
    rank_col="colbert_rank",
    top_k=5,
    model_name="princeton-nlp/sup-simcse-bert-base-uncased",
    batch_size=32
):
    """
    Compute semantic similarity between each document and top-k reference documents.
    
    For each query:
    1. Generate embeddings for all documents using SimCSE
    2. Select top-k documents as reference set (by rank_col)
    3. For each document, compute average cosine similarity with reference set
       - If document is in top-k: exclude itself from calculation
       - If document is not in top-k: compute with all top-k documents
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer and model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Compute embeddings for all documents in batches
    print("Computing embeddings for all documents...")
    texts = df[text_col].tolist()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.extend(batch_embeddings)
    df["embedding"] = list(embeddings)
    
    def cosine_sim(a, b):
        return 1 - cosine(a, b)
    
    semantic_sim_dict = {}
    
    # Group by qid and calculate semantic similarity
    for qid_val, group in df.groupby(qid_col):
        valid_rank_rows = group[group[rank_col].notna()]
        if valid_rank_rows.empty:
            group_sorted = group
        else:
            group_sorted = group.sort_values(by=rank_col, ascending=True)
            
        top_k_group = group_sorted.head(top_k)
        top_k_indices = set(top_k_group.index)
        top_k_embs = np.stack(top_k_group["embedding"].values)
        
        for idx, row in group.iterrows():
            doc_emb = row["embedding"]
            
            if idx in top_k_indices:
                # Exclude itself from reference set
                mask = [i != idx for i in top_k_group.index]
                if sum(mask) == 0:
                    avg_sim = 0.0
                else:
                    ref_embs = np.stack(top_k_group.iloc[np.where(mask)[0]]["embedding"].values)
                    sims = [cosine_sim(doc_emb, ref_emb) for ref_emb in ref_embs]
                    avg_sim = np.mean(sims)
            else:
                # Calculate similarity with all top-k
                sims = [cosine_sim(doc_emb, top_emb) for top_emb in top_k_embs]
                avg_sim = np.mean(sims)
            
            semantic_sim_dict[idx] = avg_sim
    
    df["semantic_sim"] = df.index.map(semantic_sim_dict)
    print("Semantic similarity computed and stored in 'semantic_sim'.")
    
    return df


def process_all_queries(colbert_ranking_df, sbr_ranking_df, qrels_df=None, top_k=4, 
                       model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Process all query IDs using the strategic document selection approach.
    
    Returns a DataFrame with selected documents for all queries.
    """
    # Validate data format
    colbert_required_columns = ['qid', 'query', 'docno', 'text', 'colbert_rank']
    sbr_required_columns = ['qid', 'query', 'docno', 'text', 'sbr_rank']
    
    missing_cols = [col for col in colbert_required_columns if col not in colbert_ranking_df.columns]
    if missing_cols:
        raise ValueError(f"ColBERT ranking missing columns: {', '.join(missing_cols)}")
        
    missing_cols = [col for col in sbr_required_columns if col not in sbr_ranking_df.columns]
    if missing_cols:
        raise ValueError(f"SBR ranking missing columns: {', '.join(missing_cols)}")
    
    # Validate or create dummy qrels
    if qrels_df is not None and not qrels_df.empty:
        qrels_required_columns = ['qid', 'docno', 'label']
        missing_cols = [col for col in qrels_required_columns if col not in qrels_df.columns]
        if missing_cols:
            print(f"Warning: qrels missing columns: {', '.join(missing_cols)}. Creating dummy qrels.")
            qrels_df = get_dummy_qrels_data(colbert_ranking_df, sbr_ranking_df)
    else:
        print("No qrels provided. Creating dummy qrels data.")
        qrels_df = get_dummy_qrels_data(colbert_ranking_df, sbr_ranking_df)
    
    # Get common query IDs
    all_qids = sorted(set(colbert_ranking_df['qid'].unique()) & set(sbr_ranking_df['qid'].unique()))
    
    if not all_qids:
        raise ValueError("no same qid in two rankings")
    
    print(f"Found {len(all_qids)} query IDs")
    
    # Process each query
    all_selected_docs = []
    
    for qid in all_qids:
        try:
            selected_docs = strategic_document_selection(
                colbert_ranking_df, 
                sbr_ranking_df, 
                qid, 
                qrels_df=qrels_df,
                top_k=top_k,
                model_name=model_name
            )
            all_selected_docs.append(selected_docs)
            print(f"Successfully processed ID {qid}, selected {len(selected_docs)} documents")
        except Exception as e:
            print(f"Error processing ID {qid}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not all_selected_docs:
        raise ValueError("no successful processing for any qid")
    
    # Merge all results
    result_df = pd.concat(all_selected_docs, ignore_index=True)
    
    # Print statistics
    from_counts = result_df['from'].value_counts()
    source_counts = result_df['source'].value_counts()
    
    print("\nAll selected documents distribution:")
    print(from_counts)
    print("\nAll selected documents source distribution:")
    print(source_counts)
    
    return result_df


def get_dummy_qrels_data(colbert_ranking_df, sbr_ranking_df):
    """
    Create dummy qrels assuming top-10 documents from either ranking are relevant (label=1).
    Others are non-relevant (label=0).
    """
    all_qids = sorted(set(colbert_ranking_df['qid'].unique()) & set(sbr_ranking_df['qid'].unique()))
    rows = []
    
    for qid in all_qids:
        colbert_docs = colbert_ranking_df[colbert_ranking_df['qid'] == qid].sort_values('colbert_rank')
        sbr_docs = sbr_ranking_df[sbr_ranking_df['qid'] == qid].sort_values('sbr_rank')
        
        # Top 10 from either ranking are considered relevant
        top_colbert_docnos = set(colbert_docs.head(10)['docno'])
        top_sbr_docnos = set(sbr_docs.head(10)['docno'])
        relevant_docnos = top_colbert_docnos.union(top_sbr_docnos)
        
        all_docnos = set(colbert_docs['docno']).union(set(sbr_docs['docno']))
        
        for docno in all_docnos:
            rows.append({
                'qid': qid,
                'docno': docno,
                'label': 1 if docno in relevant_docnos else 0
            })
    
    return pd.DataFrame(rows)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Strategic document selection merging ColBERT and SBR rankings"
    )
    parser.add_argument(
        "colbert_file",
        nargs="?",
        default="colbert_rankings_original.csv",
        help="Path to ColBERT (original) ranking CSV. Columns: qid, query, docno, text, colbert_rank"
    )
    parser.add_argument(
        "sbr_file",
        nargs="?",
        default="sbr_rankings_version1.csv",
        help="Path to SBR (debiased) ranking CSV. Columns: qid, query, docno, text, sbr_rank"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of top documents to select from each ranking. Default: 4"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help="Path to qrels CSV file (optional). Columns: qid, docno, label"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="strategic_selection_results_version1.csv",
        help="Output file path. Default: strategic_selection_results_version1.csv"
    )
    
    args = parser.parse_args()
    
    print("Loading ranking data...")
    
    # Load ColBERT ranking
    colbert_df = pd.read_csv(args.colbert_file)
    required_cols = ['qid', 'query', 'docno', 'text', 'colbert_rank']
    if not all(col in colbert_df.columns for col in required_cols):
        raise ValueError(f"ColBERT ranking CSV must contain columns: {required_cols}")
    
    # Load SBR ranking
    sbr_df = pd.read_csv(args.sbr_file)
    required_cols = ['qid', 'query', 'docno', 'text', 'sbr_rank']
    if not all(col in sbr_df.columns for col in required_cols):
        raise ValueError(f"SBR ranking CSV must contain columns: {required_cols}")
    
    print(f"Loaded ColBERT ranking: {len(colbert_df)} documents for {colbert_df['qid'].nunique()} queries")
    print(f"Loaded SBR ranking: {len(sbr_df)} documents for {sbr_df['qid'].nunique()} queries")
    
    # Load qrels if provided
    qrels_df = None
    if args.qrels:
        qrels_df = pd.read_csv(args.qrels)
        print(f"Loaded qrels: {len(qrels_df)} judgments")
    
    # Process all queries
    print(f"\nProcessing with top_k={args.top_k}...")
    result_df = process_all_queries(
        colbert_df, 
        sbr_df, 
        qrels_df=qrels_df,
        top_k=args.top_k,
        model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    )
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    result_df.to_csv(args.output, index=False)
    
    print("\n=== Processing Complete ===")
    print(f"Total selected documents: {len(result_df)}")
    print(f"Queries processed: {result_df['qid'].nunique()}")
    print(f"Average documents per query: {len(result_df) / result_df['qid'].nunique():.1f}")


if __name__ == "__main__":
    main()
