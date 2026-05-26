"""
Strategic Document Selection - Version 2
Merges SBR and IPSsimRF rankings for User Study 2 evaluation.

This script selects documents from both SBR (from Step 1) and IPSsimRF (from Step 5)
rankings to create a balanced evaluation set for comparing the two methods.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import argparse


def add_semantic_similarity_hf(df, qid_col="qid", text_col="text", rank_col="rank",
                               top_k=5, model_name="princeton-nlp/sup-simcse-bert-base-uncased",
                               batch_size=32):
    """
    Compute semantic similarity using SimCSE embeddings.
    
    Args:
        df: DataFrame with documents
        qid_col: Query ID column
        text_col: Document text column
        rank_col: Ranking column to identify top-k documents
        top_k: Number of top documents for reference
        model_name: SimCSE model name
        batch_size: Batch size for embedding computation
    
    Returns:
        DataFrame with added 'semantic_sim' column
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer and model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
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
    
    # Compute similarity for each query
    semantic_sim_dict = {}
    
    for qid_val, group in df.groupby(qid_col):
        group_sorted = group.sort_values(rank_col) if pd.notna(rank_col) and rank_col in group.columns else group
        top_k_group = group_sorted.head(top_k)
        top_k_indices = set(top_k_group.index)
        top_k_embs = np.stack(top_k_group["embedding"].values)
        
        for idx, row in group.iterrows():
            doc_emb = row["embedding"]
            
            if idx in top_k_indices:
                # Exclude self from top-k
                mask = [i != idx for i in top_k_group.index]
                if sum(mask) > 0:
                    ref_embs = np.stack(top_k_group.iloc[np.where(mask)[0]]["embedding"].values)
                    sims = [1 - cosine(doc_emb, ref_emb) for ref_emb in ref_embs]
                    avg_sim = np.mean(sims)
                else:
                    avg_sim = 0.0
            else:
                # Not in top-k, compare with all top-k
                sims = [1 - cosine(doc_emb, top_emb) for top_emb in top_k_embs]
                avg_sim = np.mean(sims)
            
            semantic_sim_dict[idx] = avg_sim
    
    df["semantic_sim"] = df.index.map(semantic_sim_dict)
    df.drop(columns=["embedding"], inplace=True)
    
    print("Semantic similarity computed and stored in 'semantic_sim'.")
    return df


def strategic_document_selection(sbr_ranking_df, ipssimrf_ranking_df, qid, qrels_df=None, 
                                 top_k=4, model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Strategic document selection from SBR and IPSsimRF rankings.
    
    Args:
        sbr_ranking_df: DataFrame with SBR rankings
        ipssimrf_ranking_df: DataFrame with IPSsimRF rankings
        qid: Query ID
        qrels_df: Relevance judgments (optional)
        top_k: Number of documents from each ranking
        model_name: SimCSE model name
    
    Returns:
        DataFrame with selected documents
    """
    # Filter for this query
    sbr_df = sbr_ranking_df[sbr_ranking_df['qid'] == qid].copy()
    ipssimrf_df = ipssimrf_ranking_df[ipssimrf_ranking_df['qid'] == qid].copy()
    
    if sbr_df.empty or ipssimrf_df.empty:
        raise ValueError(f"qid={qid} is missing in one/both rankings")
    
    query_text = sbr_df['query'].iloc[0]
    
    # Sort by rank
    sbr_sorted = sbr_df.sort_values('sbr_rank').reset_index(drop=True)
    ipssimrf_sorted = ipssimrf_df.sort_values('ipssimrf_rank').reset_index(drop=True)
    
    selected_docs = []
    selected_docnos = set()
    
    # Step 1: Select top-k from SBR ranking
    for i in range(min(top_k, len(sbr_sorted))):
        doc = sbr_sorted.iloc[i]
        
        # Find IPSsimRF rank
        ipssimrf_rank = np.nan
        matching_ipssimrf = ipssimrf_df[ipssimrf_df['docno'] == doc['docno']]
        if not matching_ipssimrf.empty:
            ipssimrf_rank = matching_ipssimrf.iloc[0]['ipssimrf_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'sbr_rank': doc['sbr_rank'],
            'ipssimrf_rank': ipssimrf_rank,
            'source': "top from sbr",
            'from': "sbr",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
    
    # Step 2: Select top-k from IPSsimRF ranking (skipping duplicates)
    ipssimrf_idx = 0
    selected_from_ipssimrf = 0
    
    while selected_from_ipssimrf < top_k and ipssimrf_idx < len(ipssimrf_sorted):
        doc = ipssimrf_sorted.iloc[ipssimrf_idx]
        ipssimrf_idx += 1
        
        if doc['docno'] in selected_docnos:
            continue
        
        # Find SBR rank
        sbr_rank = np.nan
        matching_sbr = sbr_df[sbr_df['docno'] == doc['docno']]
        if not matching_sbr.empty:
            sbr_rank = matching_sbr.iloc[0]['sbr_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'sbr_rank': sbr_rank,
            'ipssimrf_rank': doc['ipssimrf_rank'],
            'source': "top from ipssimrf",
            'from': "ipssimrf",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
        selected_from_ipssimrf += 1
    
    # Step 3: Add easy negative (low similarity, label=0)
    remaining_docs = ipssimrf_sorted[~ipssimrf_sorted['docno'].isin(selected_docnos)].copy()
    
    if remaining_docs.empty:
        return pd.DataFrame(selected_docs)
    
    # Compute semantic similarity
    temp_df = pd.DataFrame(selected_docs)
    all_docs_df = pd.concat([remaining_docs, temp_df]).reset_index(drop=True)
    
    try:
        docs_with_sim = add_semantic_similarity_hf(
            all_docs_df,
            qid_col="qid",
            text_col="text",
            rank_col="ipssimrf_rank",
            top_k=top_k,
            model_name=model_name,
            batch_size=32
        )
        
        remaining_with_sim = docs_with_sim[docs_with_sim['docno'].isin(remaining_docs['docno'])].copy()
        
        # Filter for label=0 if qrels available
        if qrels_df is not None:
            query_qrels = qrels_df[qrels_df['qid'] == qid]
            remaining_with_sim['label'] = remaining_with_sim['docno'].apply(
                lambda x: query_qrels[query_qrels['docno'] == x]['label'].iloc[0]
                if x in query_qrels['docno'].values else 0
            )
            negative_docs = remaining_with_sim[remaining_with_sim['label'] == 0]
        else:
            negative_docs = remaining_with_sim
        
        if not negative_docs.empty:
            easy_neg = negative_docs.nsmallest(1, 'semantic_sim').iloc[0]
            
            sbr_rank = np.nan
            matching_sbr = sbr_df[sbr_df['docno'] == easy_neg['docno']]
            if not matching_sbr.empty:
                sbr_rank = matching_sbr.iloc[0]['sbr_rank']
            
            selected_docs.append({
                'qid': qid,
                'query': query_text,
                'docno': easy_neg['docno'],
                'text': easy_neg['text'],
                'sbr_rank': sbr_rank,
                'ipssimrf_rank': easy_neg.get('ipssimrf_rank', np.nan),
                'semantic_sim': easy_neg['semantic_sim'],
                'source': "easy negative",
                'from': "ipssimrf",
                'selected_in_turn': len(selected_docs) + 1
            })
    
    except Exception as e:
        print(f"Error computing semantic similarity for qid {qid}: {e}")
    
    return pd.DataFrame(selected_docs)


def get_qrels_data():
    """Load TREC DL 2021 qrels using PyTerrier."""
    import pyterrier as pt
    if not pt.started():
        pt.init()
    
    dataset = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2021/judged')
    qrels = dataset.get_qrels()
    return qrels


def process_all_queries(sbr_ranking_df, ipssimrf_ranking_df, qrels_df=None, top_k=4,
                       model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Process all queries and select documents strategically.
    
    Args:
        sbr_ranking_df: SBR ranking results
        ipssimrf_ranking_df: IPSsimRF ranking results
        qrels_df: Relevance judgments
        top_k: Documents to select from each ranking
        model_name: SimCSE model name
    
    Returns:
        DataFrame with all selected documents
    """
    all_qids = set(sbr_ranking_df['qid'].unique()) & set(ipssimrf_ranking_df['qid'].unique())
    all_selected_docs = []
    
    for qid in sorted(all_qids):
        try:
            selected_df = strategic_document_selection(
                sbr_ranking_df, 
                ipssimrf_ranking_df, 
                qid, 
                qrels_df=qrels_df,
                top_k=top_k,
                model_name=model_name
            )
            
            # Add labels if qrels available
            if qrels_df is not None and 'label' not in selected_df.columns:
                query_qrels = qrels_df[qrels_df['qid'] == qid]
                selected_df['label'] = selected_df['docno'].apply(
                    lambda x: query_qrels[query_qrels['docno'] == x]['label'].iloc[0]
                    if x in query_qrels['docno'].values else 0
                )
            
            all_selected_docs.append(selected_df)
            print(f"Successfully processed ID {qid}, selected {len(selected_df)} documents")
            
        except Exception as e:
            print(f"Error processing qid {qid}: {e}")
            continue
    
    result_df = pd.concat(all_selected_docs, ignore_index=True)
    
    print("\nAll selected documents distribution:")
    print(result_df['from'].value_counts())
    print("\nAll selected documents source distribution:")
    print(result_df['source'].value_counts())
    
    return result_df


def main(sbr_file, ipssimrf_file, output_file="strategic_selection_results_version2.csv",
         top_k=4, qrels_file=None):
    """
    Main workflow for strategic document selection (Version 2).
    
    Args:
        sbr_file: SBR ranking CSV (from Step 1)
        ipssimrf_file: IPSsimRF ranking CSV (from Step 5)
        output_file: Output CSV path
        top_k: Documents to select from each ranking
        qrels_file: Optional qrels CSV file
    """
    print("=" * 60)
    print("Strategic Document Selection - Version 2")
    print("Merging SBR and IPSsimRF rankings for User Study 2")
    print("=" * 60)
    
    # Load rankings
    print(f"\nLoading SBR rankings from: {sbr_file}")
    sbr_df = pd.read_csv(sbr_file, lineterminator="\n")
    
    print(f"Loading IPSsimRF rankings from: {ipssimrf_file}")
    ipssimrf_df = pd.read_csv(ipssimrf_file, lineterminator="\n")
    
    # Ensure required columns exist
    required_cols_sbr = ['qid', 'query', 'docno', 'text', 'sbr_rank']
    required_cols_ipssimrf = ['qid', 'query', 'docno', 'text']
    
    # Check for rank column in IPSsimRF (could be ipssimrf_rank or unbiased_rank)
    if 'ipssimrf_rank' not in ipssimrf_df.columns and 'unbiased_rank' in ipssimrf_df.columns:
        ipssimrf_df = ipssimrf_df.rename(columns={'unbiased_rank': 'ipssimrf_rank'})
    
    required_cols_ipssimrf.append('ipssimrf_rank')
    
    # Load qrels
    if qrels_file:
        print(f"\nLoading qrels from: {qrels_file}")
        qrels_df = pd.read_csv(qrels_file)
    else:
        print("\nLoading qrels from PyTerrier dataset...")
        qrels_df = get_qrels_data()
    
    print(f"Loaded {len(qrels_df)} relevance judgments")
    
    # Process all queries
    print("\nProcessing all queries...")
    result_df = process_all_queries(
        sbr_df,
        ipssimrf_df,
        qrels_df=qrels_df,
        top_k=top_k,
        model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    )
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("Strategic Selection Complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_file}")
    print(f"Total queries: {result_df['qid'].nunique()}")
    print(f"Total documents selected: {len(result_df)}")
    print(f"Average documents per query: {len(result_df) / result_df['qid'].nunique():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategic document selection for User Study 2 (SBR vs IPSsimRF)"
    )
    
    parser.add_argument(
        "sbr_file",
        type=str,
        nargs="?",
        default="sbr_rankings.csv",
        help="Path to SBR ranking CSV (from Step 1)"
    )
    
    parser.add_argument(
        "ipssimrf_file",
        type=str,
        nargs="?",
        default="ipssimrf_rankings.csv",
        help="Path to IPSsimRF ranking CSV (from Step 5)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="strategic_selection_results_version2.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of documents to select from each ranking (default: 4)"
    )
    
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help="Optional qrels file path (uses PyTerrier dataset if not provided)"
    )
    
    args = parser.parse_args()
    
    main(
        sbr_file=args.sbr_file,
        ipssimrf_file=args.ipssimrf_file,
        output_file=args.output,
        top_k=args.top_k,
        qrels_file=args.qrels
    )
