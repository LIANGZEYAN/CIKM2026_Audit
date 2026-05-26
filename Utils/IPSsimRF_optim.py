"""
IPSsimRF: Iterative Refinement via Audit Signal
This script optimizes the alpha parameter to balance propensity scores with semantic similarity
based on user click feedback (audit signals) from User Study 1.

Key Formula (from paper Section 3.3):
Score_IPSsimRF(d) = Score_ColBERT(d) * (1/ρd + α * AvgSim(d, D_top))

where:
- Score_ColBERT(d): Initial relevance score from ColBERT
- ρd: Propensity score (position bias)
- AvgSim(d, D_top): Average semantic similarity to top-ranked documents
- α: Hyperparameter tuned using audit signals (user clicks)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import os


def remove_duplicates(df, qid_col="qid", text_col="text", rank_col="rank"):
    """
    Remove duplicate documents within each query based on normalized text content.
    Keeps the first occurrence and reassigns consecutive rank positions.
    
    Args:
        df: DataFrame with at least [qid, text, rank] columns
        qid_col: Query ID column name (default "qid")
        text_col: Text column for duplicate detection (default "text")
        rank_col: Original ranking column (default "rank")
    
    Returns:
        Deduplicated DataFrame with reassigned ranks
    """
    df = df.copy()
    df["_temp_index"] = range(len(df))
    df.sort_values(by=[qid_col, "_temp_index"], ascending=[True, True], inplace=True)
    
    duplicate_counts = {}
    list_of_subdfs = []
    
    # Remove duplicates within each query
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
    
    # Concatenate deduplicated dataframes
    dedup_df = pd.concat(list_of_subdfs, ignore_index=True)
    dedup_df.sort_values(by="_temp_index", ascending=True, inplace=True)
    
    # Reassign consecutive ranks within each query
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
    print(f"Before deduplication: {len(df)} rows, After: {len(final_df)} rows")
    
    return final_df


def load_real_clicks(file_path):
    """
    Load real user click data from User Study evaluation interface.
    Each row represents a click event.
    
    Args:
        file_path: Path to CSV file containing real click logs
    
    Returns:
        DataFrame with columns [qid, docno, real_clicks] where real_clicks is click count
    """
    print(f"Loading click logs from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        
        # Handle different column naming conventions
        if 'qid' not in df.columns and 'query_id' in df.columns:
            df['qid'] = df['query_id']
        if 'docno' not in df.columns and 'doc_id' in df.columns:
            df['docno'] = df['doc_id']
        
        # Ensure correct data types
        df['qid'] = df['qid'].astype(str)
        df['docno'] = df['docno'].astype(str)
        
        # Count clicks for each query-document pair
        click_counts = df.groupby(['qid', 'docno']).size().reset_index(name='real_clicks')
        
        # Print statistics
        print(f"Loaded {len(df)} click events")
        print(f"Covering {len(click_counts)} unique query-document pairs")
        print(f"Covering {click_counts['qid'].nunique()} unique queries")
        print(f"Average documents clicked per query: {click_counts.groupby('qid')['docno'].count().mean():.2f}")
        
        return click_counts
    
    except Exception as e:
        print(f"Error loading click data: {e}")
        return pd.DataFrame(columns=['qid', 'docno', 'real_clicks'])


def add_semantic_similarity(
    df,
    qid_col="qid",
    text_col="text",
    rank_col="colbert_rank",
    top_k=5,
    model_name="princeton-nlp/sup-simcse-bert-base-uncased",
    batch_size=32
):
    """
    Compute semantic similarity for each document relative to top-k documents.
    Uses SimCSE embeddings from Hugging Face.
    
    Args:
        df: Input DataFrame with at least [qid, text, rank_col] columns
        qid_col: Query ID column name (default "qid")
        text_col: Document text column (default "text")
        rank_col: Ranking column for determining top-k (default "colbert_rank")
        top_k: Number of top documents for reference (default 5)
        model_name: SimCSE model name (default "princeton-nlp/sup-simcse-bert-base-uncased")
        batch_size: Batch size for embedding computation (default 32)
    
    Returns:
        DataFrame with added columns "embedding" and "semantic_sim"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading SimCSE tokenizer and model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Compute embeddings for all documents
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
    
    # Define cosine similarity function
    def cosine_sim(a, b):
        return 1 - cosine(a, b)
    
    semantic_sim_dict = {}
    
    # Compute semantic similarity for each document within its query
    for qid_val, group in df.groupby(qid_col):
        group_sorted = group.sort_values(by=rank_col, ascending=True)
        top_k_group = group_sorted.head(top_k)
        top_k_indices = set(top_k_group.index)
        top_k_embs = np.stack(top_k_group["embedding"].values)
        
        for idx, row in group.iterrows():
            doc_emb = row["embedding"]
            
            # If document is in top-k, exclude itself from reference set
            if idx in top_k_indices:
                mask = [i != idx for i in top_k_group.index]
                if sum(mask) == 0:
                    avg_sim = 0.0
                else:
                    ref_embs = np.stack(top_k_group.iloc[np.where(mask)[0]]["embedding"].values)
                    sims = [cosine_sim(doc_emb, ref_emb) for ref_emb in ref_embs]
                    avg_sim = np.mean(sims)
            else:
                # Document not in top-k, compute similarity to all top-k documents
                sims = [cosine_sim(doc_emb, top_emb) for top_emb in top_k_embs]
                avg_sim = np.mean(sims)
            
            semantic_sim_dict[idx] = avg_sim
    
    df["semantic_sim"] = df.index.map(semantic_sim_dict)
    
    print("Semantic similarity computed and stored in 'semantic_sim' column.")
    return df


def analyze_rd_distribution(colbert_rankings, real_clicks, display_docs_path=None):
    """
    Analyze Rank Deficit (RD) distribution (paper Section 3.3, Eq. 3).
    RD measures how actual click rates deviate from expected rates based on ColBERT rankings.
    
    Args:
        colbert_rankings: DataFrame with ColBERT scores and CTR estimates
        real_clicks: DataFrame with real user click counts
        display_docs_path: Optional path to CSV with documents shown to users
    
    Returns:
        DataFrame with RD analysis results including orig_rank column
    """
    df = colbert_rankings.copy()
    clicks = real_clicks.copy()
    
    # Convert ID columns to strings
    df['qid'] = df['qid'].astype(str)
    df['docno'] = df['docno'].astype(str)
    clicks['qid'] = clicks['qid'].astype(str)
    clicks['docno'] = clicks['docno'].astype(str)
    
    # If display document info is provided, filter to only those documents
    if display_docs_path is not None:
        print(f"Loading display document information from: {display_docs_path}")
        display_docs = pd.read_csv(display_docs_path)
        display_docs['qid'] = display_docs['qid'].astype(str)
        display_docs['docno'] = display_docs['docno'].astype(str)
        
        # Keep only documents that were displayed to users
        df = pd.merge(
            df,
            display_docs[['qid', 'docno']],
            on=['qid', 'docno'],
            how='inner'
        )
    
    # Merge with real clicks
    merged = pd.merge(df, clicks, on=['qid', 'docno'], how='left')
    merged['real_clicks'] = merged['real_clicks'].fillna(0)
    
    # Compute original ranking positions (based on CTR/score)
    merged['orig_rank'] = merged.groupby('qid')['ctr'].rank(method='first', ascending=False).astype(int)
    
    # Calculate real CTR
    merged['real_ctr'] = merged.groupby('qid')['real_clicks'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    
    # Calculate CTR ratio (normalized within query)
    merged['ctr_ratio'] = merged.groupby('qid')['ctr'].transform(
        lambda x: x / (x.sum() + 1e-10)
    )
    
    # Calculate RD (Click-Through Rate Ratio)
    # RD > 1 means document received more clicks than expected
    # RD < 1 means document received fewer clicks than expected
    merged['rd'] = merged['real_ctr'] / (merged['ctr_ratio'] + 1e-10)
    
    # Calculate query-level statistics
    query_click_stats = merged.groupby('qid').agg(
        total_docs=('docno', 'count'),
        clicked_docs=('real_clicks', lambda x: (x > 0).sum()),
        click_ratio=('real_clicks', lambda x: (x > 0).mean())
    ).reset_index()
    
    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Total queries: {merged['qid'].nunique()}")
    print(f"Total displayed documents: {len(merged)}")
    print(f"Average documents displayed per query: {merged.groupby('qid')['docno'].count().mean():.2f}")
    print(f"Queries with clicks: {query_click_stats[query_click_stats['clicked_docs'] > 0].shape[0]}")
    print(f"Documents with clicks: {(merged['real_clicks'] > 0).sum()}")
    
    print("\n=== Click Distribution ===")
    print(f"Average documents clicked per query: {query_click_stats['clicked_docs'].mean():.2f}")
    print(f"Click rate: {query_click_stats['click_ratio'].mean():.2%}")
    
    print("\n=== RD Statistics ===")
    print(f"RD mean: {merged['rd'].mean():.4f}")
    print(f"RD median: {merged['rd'].median():.4f}")
    print(f"Proportion with RD > 1: {(merged['rd'] > 1).mean():.2%}")
    print(f"Proportion with RD > 2: {(merged['rd'] > 2).mean():.2%}")
    
    # Create visualizations
    plt.figure(figsize=(15, 12))
    
    # RD distribution
    plt.subplot(2, 2, 1)
    sns.histplot(merged['rd'].clip(0, 5), bins=50)
    plt.title('RD Distribution (clipped at 5)')
    plt.xlabel('RD')
    plt.ylabel('Number of Documents')
    plt.axvline(x=1.0, color='red', linestyle='--', label='RD = 1')
    plt.legend()
    
    # RD vs original rank
    plt.subplot(2, 2, 2)
    rank_data = merged[merged['orig_rank'] <= 20].copy()
    sns.boxplot(x='orig_rank', y='rd', data=rank_data)
    plt.title('RD Distribution by Rank Position (Top 20)')
    plt.xlabel('Original Rank')
    plt.ylabel('RD')
    plt.axhline(y=1.0, color='red', linestyle='--')
    
    # Expected CTR vs Real CTR
    plt.subplot(2, 2, 3)
    plt.scatter(merged['ctr'], merged['real_ctr'], alpha=0.5)
    plt.title('Expected CTR vs Real CTR')
    plt.xlabel('Expected CTR (from ColBERT)')
    plt.ylabel('Real CTR (from users)')
    max_val = max(merged['ctr'].max(), merged['real_ctr'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    plt.legend()
    
    # Semantic similarity vs RD (if available)
    if 'semantic_sim' in merged.columns:
        plt.subplot(2, 2, 4)
        plt.scatter(merged['semantic_sim'], merged['rd'], alpha=0.5)
        plt.title('Semantic Similarity vs RD')
        plt.xlabel('Semantic Similarity')
        plt.ylabel('RD')
        plt.axhline(y=1.0, color='red', linestyle='--')
        
        corr = merged['semantic_sim'].corr(merged['rd'])
        plt.annotate(f'Correlation: {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig('rd_distribution.png', dpi=300)
    print("\nVisualization saved to 'rd_distribution.png'")
    
    # Analyze high vs low RD documents
    high_rd = merged[merged['rd'] > 1.0]
    low_rd = merged[merged['rd'] <= 1.0]
    
    print("\n=== High RD Document Analysis ===")
    print(f"High RD documents: {len(high_rd)}")
    print(f"Percentage of high RD: {len(high_rd) / len(merged):.2%}")
    print(f"Average rank of high RD docs: {high_rd['orig_rank'].mean():.2f}")
    print(f"Average rank of low RD docs: {low_rd['orig_rank'].mean():.2f}")
    
    if 'semantic_sim' in merged.columns:
        print(f"Average semantic similarity (high RD): {high_rd['semantic_sim'].mean():.4f}")
        print(f"Average semantic similarity (low RD): {low_rd['semantic_sim'].mean():.4f}")
    
    return merged


def optimize_alpha_ipssimrf(
    df,
    rd_threshold=1.0,
    alpha_range=(0.1, 3.0),
    max_alpha=2.0,
    steps=30,
    stability_weight=0.2,
    stability_cap=0.2
):
    """
    Optimize alpha parameter for IPSsimRF using audit signals (RD from real clicks).
    
    The optimization aims to:
    1. Increase rankings of high RD documents (underestimated by ColBERT)
    2. Decrease rankings of low RD documents (overestimated by ColBERT)
    3. Maintain reasonable ranking stability
    
    Uses the paper formula (Equation 2):
        Score_IPSsimRF(d) = Score_ColBERT(d) × (1/ρd + α × AvgSim(d, D_top))
    
    Where ρd is the position-based propensity score.
    
    Args:
        df: DataFrame with RD, CTR, semantic similarity, and rank columns
        rd_threshold: Threshold for high/low RD classification
        alpha_range: Alpha search range (min_value, max_value)
        max_alpha: Maximum allowed alpha value
        steps: Number of alpha values to test
        stability_weight: Weight for stability in scoring (0-1)
        stability_cap: Maximum stability value (prevents over-weighting stability)
    
    Returns:
        Tuple of (best_alpha, optimized_dataframe)
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    print(f"\nStarting alpha optimization for IPSsimRF")
    print(f"Search range: {alpha_range}, max_alpha: {max_alpha}")
    print(f"Stability weight: {stability_weight}, stability cap: {stability_cap}")
    
    # Verify required columns
    required_cols = ['ctr', 'semantic_sim', 'rd', 'orig_rank']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df_copy = df.copy()
    
    # Normalize CTR if needed
    if df_copy['ctr'].max() > 1.0 or df_copy['ctr'].min() < 0.0:
        print("Note: Normalizing CTR column")
        df_copy['normalized_ctr'] = df_copy.groupby('qid')['ctr'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )
    else:
        df_copy['normalized_ctr'] = df_copy['ctr']
    
    # Compute position-based propensity scores
    # Higher positions have higher propensity (more likely to be clicked due to position bias)
    df_copy['propensity'] = df_copy['orig_rank'].apply(
        lambda k: 1.0 / (1.0 + np.log1p(k))
    )
    
    print(f"\nPropensity score statistics:")
    print(f"  Min (lowest position): {df_copy['propensity'].min():.4f}")
    print(f"  Max (highest position): {df_copy['propensity'].max():.4f}")
    print(f"  Mean: {df_copy['propensity'].mean():.4f}")
    
    # Identify high and low RD documents
    df_copy['is_high_rd'] = df_copy['rd'] > rd_threshold
    
    high_rd_count = df_copy['is_high_rd'].sum()
    low_rd_count = (~df_copy['is_high_rd']).sum()
    
    print(f"\nHigh RD documents (RD > {rd_threshold}): {high_rd_count}")
    print(f"Low RD documents (RD <= {rd_threshold}): {low_rd_count}")
    
    # Test different alpha values
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], steps)
    results = []
    
    for alpha_value in tqdm(alpha_values, desc="Testing alpha values"):
        # Apply IPSsimRF formula from paper (Equation 2):
        # Score_IPSsimRF(d) = Score_ColBERT(d) × (1/ρd + α × AvgSim(d, D_top))
        adjusted_scores = df_copy['normalized_ctr'] * (
            (1.0 / df_copy['propensity']) + alpha_value * df_copy['semantic_sim']
        )
        
        df_copy['temp_adjusted_score'] = adjusted_scores
        
        # Evaluate performance
        high_low_diff = 0
        rank_changes = []
        
        for qid in df_copy['qid'].unique():
            query_docs = df_copy[df_copy['qid'] == qid].copy()
            
            if len(query_docs) < 2:
                continue
            
            # Calculate original and new ranks
            query_docs['orig_rank_in_query'] = query_docs['normalized_ctr'].rank(ascending=False)
            query_docs['new_rank'] = query_docs['temp_adjusted_score'].rank(ascending=False)
            query_docs['rank_change'] = query_docs['orig_rank_in_query'] - query_docs['new_rank']
            
            # Calculate average rank changes for high and low RD documents
            high_rd_query = query_docs[query_docs['is_high_rd']]
            low_rd_query = query_docs[~query_docs['is_high_rd']]
            
            if len(high_rd_query) > 0 and len(low_rd_query) > 0:
                high_change = high_rd_query['rank_change'].mean()
                low_change = low_rd_query['rank_change'].mean()
                # High RD should increase (positive), low RD should decrease (negative)
                high_low_diff += high_change - low_change
            
            rank_changes.extend(query_docs['rank_change'].abs().tolist())
        
        # Calculate ranking stability (smaller changes = more stable)
        avg_rank_change = np.mean(rank_changes) if rank_changes else 0
        rank_stability = 1 / (1 + avg_rank_change)
        rank_stability = min(rank_stability, stability_cap)
        
        results.append({
            'alpha': alpha_value,
            'high_low_diff': high_low_diff,
            'raw_stability': rank_stability,
            'capped_stability': rank_stability
        })
    
    # Remove temporary column
    if 'temp_adjusted_score' in df_copy.columns:
        df_copy.drop('temp_adjusted_score', axis=1, inplace=True)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal alpha
    if len(results_df) > 0:
        # Normalize metrics
        max_diff = results_df['high_low_diff'].max()
        if max_diff > 0:
            results_df['norm_diff'] = results_df['high_low_diff'] / max_diff
        else:
            results_df['norm_diff'] = 0
        
        # Compute composite score (balance difference and stability)
        results_df['score'] = (
            (1 - stability_weight) * results_df['norm_diff'] +
            stability_weight * results_df['capped_stability']
        )
        
        # Filter results within max_alpha constraint
        valid_results = results_df[results_df['alpha'] <= max_alpha]
        
        if len(valid_results) > 0:
            best_idx = valid_results['score'].idxmax()
            best_alpha = valid_results.loc[best_idx, 'alpha']
            best_results = valid_results.loc[best_idx]
        else:
            best_alpha = max_alpha
            best_results = results_df[results_df['alpha'] <= max_alpha].iloc[-1]
            print(f"Warning: No optimal alpha found. Using max_alpha: {max_alpha}")
    else:
        best_alpha = alpha_range[0]
        best_results = pd.Series({
            'high_low_diff': 0,
            'raw_stability': 1.0,
            'capped_stability': stability_cap,
            'score': 0
        })
    
    # Calculate final scores with best alpha using paper formula
    df_copy['ipssimrf_score'] = df_copy['normalized_ctr'] * (
        (1.0 / df_copy['propensity']) + best_alpha * df_copy['semantic_sim']
    )
    
    # Calculate new ranks
    df_copy['orig_rank_in_query'] = df_copy.groupby('qid')['normalized_ctr'].rank(
        method='first', ascending=False
    )
    df_copy['new_rank'] = df_copy.groupby('qid')['ipssimrf_score'].rank(
        method='first', ascending=False
    )
    df_copy['rank_change'] = df_copy['orig_rank_in_query'] - df_copy['new_rank']
    
    # Print results
    print("\n=== Optimal Alpha Value ===")
    print(f"Alpha: {best_alpha:.4f}")
    print(f"High-Low RD rank difference: {best_results['high_low_diff']:.4f}")
    print(f"Raw rank stability: {best_results['raw_stability']:.4f}")
    print(f"Capped rank stability: {best_results['capped_stability']:.4f}")
    print(f"Composite score: {best_results['score']:.4f}")
    
    # Create optimization visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Alpha vs high-low difference
    axes[0, 0].plot(results_df['alpha'], results_df['high_low_diff'], 'b-')
    axes[0, 0].axvline(x=best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.2f}')
    axes[0, 0].set_xlabel('Alpha')
    axes[0, 0].set_ylabel('High-Low RD Rank Difference')
    axes[0, 0].set_title('Alpha vs Rank Difference')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha vs stability
    axes[0, 1].plot(results_df['alpha'], results_df['capped_stability'], 'g-')
    axes[0, 1].axvline(x=best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.2f}')
    axes[0, 1].set_xlabel('Alpha')
    axes[0, 1].set_ylabel('Rank Stability')
    axes[0, 1].set_title('Alpha vs Stability')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alpha vs composite score
    axes[1, 0].plot(results_df['alpha'], results_df['score'], 'purple')
    axes[1, 0].axvline(x=best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.2f}')
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].set_ylabel('Composite Score')
    axes[1, 0].set_title('Alpha vs Composite Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rank changes for high vs low RD
    high_rd_change = df_copy[df_copy['is_high_rd']]['rank_change'].mean()
    low_rd_change = df_copy[~df_copy['is_high_rd']]['rank_change'].mean()
    
    axes[1, 1].bar(['High RD', 'Low RD'], [high_rd_change, low_rd_change])
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_ylabel('Average Rank Change')
    axes[1, 1].set_title(f'Rank Changes at α={best_alpha:.2f}')
    
    for i, v in enumerate([high_rd_change, low_rd_change]):
        axes[1, 1].text(i, v + 0.1 if v > 0 else v - 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('alpha_optimization_ipssimrf.png', dpi=300)
    print("\nOptimization visualization saved to 'alpha_optimization_ipssimrf.png'")
    
    return best_alpha, df_copy


def main(colbert_rankings_path, real_clicks_path, display_docs_path=None, 
         top_k=5, rd_threshold=1.5, alpha_range=(0.5, 3.0), max_alpha=3.0):
    """
    Main workflow for IPSsimRF optimization.
    
    Args:
        colbert_rankings_path: Path to ColBERT ranking results CSV
        real_clicks_path: Path to real user click logs CSV
        display_docs_path: Optional path to displayed documents CSV
        top_k: Number of top documents for semantic similarity
        rd_threshold: Threshold for high/low RD classification
        alpha_range: Range for alpha parameter search
        max_alpha: Maximum allowed alpha value
    """
    # Step 1: Load and preprocess ColBERT rankings
    print("Step 1: Loading ColBERT rankings...")
    df_colbert = pd.read_csv(colbert_rankings_path, lineterminator="\n")
    
    # Remove duplicates
    df_colbert = remove_duplicates(
        df_colbert,
        qid_col="qid",
        text_col="text",
        rank_col="rank"
    )
    
    # Calculate CTR from ColBERT scores (simulated propensity)
    df_colbert["ctr"] = df_colbert.groupby("qid")["score"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    df_colbert["colbert_rank"] = (
        df_colbert.groupby("qid")["ctr"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    
    # Step 2: Load real user clicks
    print("\nStep 2: Loading real user clicks...")
    real_clicks = load_real_clicks(real_clicks_path)
    
    # Step 3: Analyze RD
    print("\nStep 3: Analyzing RD distribution...")
    results = analyze_rd_distribution(
        colbert_rankings=df_colbert,
        real_clicks=real_clicks,
        display_docs_path=display_docs_path
    )
    
    # Step 4: Add semantic similarity
    print("\nStep 4: Computing semantic similarity...")
    results_with_sim = add_semantic_similarity(
        results,
        qid_col="qid",
        text_col="text",
        rank_col="colbert_rank",
        top_k=top_k,
        model_name="princeton-nlp/sup-simcse-bert-base-uncased",
        batch_size=32
    )
    
    # Step 5: Optimize alpha parameter
    print("\nStep 5: Optimizing alpha parameter...")
    best_alpha, optimized_results = optimize_alpha_ipssimrf(
        results_with_sim,
        rd_threshold=rd_threshold,
        alpha_range=alpha_range,
        max_alpha=max_alpha,
        steps=50,
        stability_weight=0.5,
        stability_cap=0.7
    )
    
    # Step 6: Save results
    output_path = 'ipssimrf_optimized_results.csv'
    optimized_results.to_csv(output_path, index=False)
    print(f"\nOptimization complete! Results saved to '{output_path}'")
    print(f"Best alpha value: {best_alpha:.4f}")
    
    # Step 7: Analyze optimization results
    print("\n=== Optimization Impact Analysis ===")
    
    high_rd = optimized_results[optimized_results['is_high_rd']]
    low_rd = optimized_results[~optimized_results['is_high_rd']]
    
    orig_high_avg_rank = high_rd['orig_rank'].mean()
    new_high_avg_rank = high_rd['new_rank'].mean()
    high_rank_improvement = orig_high_avg_rank - new_high_avg_rank
    
    orig_low_avg_rank = low_rd['orig_rank'].mean()
    new_low_avg_rank = low_rd['new_rank'].mean()
    low_rank_change = orig_low_avg_rank - new_low_avg_rank
    
    print(f"\nHigh RD documents (RD > {rd_threshold}):")
    print(f"  Count: {len(high_rd)}")
    print(f"  Average original rank: {orig_high_avg_rank:.2f}")
    print(f"  Average new rank: {new_high_avg_rank:.2f}")
    print(f"  Average rank improvement: {high_rank_improvement:.2f}")
    
    print(f"\nLow RD documents (RD <= {rd_threshold}):")
    print(f"  Count: {len(low_rd)}")
    print(f"  Average original rank: {orig_low_avg_rank:.2f}")
    print(f"  Average new rank: {new_low_avg_rank:.2f}")
    print(f"  Average rank change: {low_rank_change:.2f}")
    
    return best_alpha, optimized_results


if __name__ == "__main__":
    # Example usage
    colbert_rankings_path = "colbert_rankings.csv"
    real_clicks_path = "user_clicks.csv"
    display_docs_path = "displayed_documents.csv"  # Optional
    
    best_alpha, results = main(
        colbert_rankings_path=colbert_rankings_path,
        real_clicks_path=real_clicks_path,
        display_docs_path=display_docs_path,
        top_k=5,
        rd_threshold=1.5,
        alpha_range=(0.5, 3.0),
        max_alpha=3.0
    )
