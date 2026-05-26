"""
Evaluate IPSsimRF Rankings with nDCG Metrics

This script evaluates the IPSsimRF ranking results using PyTerrier.
Simply run this script with the provided CSV file to reproduce the nDCG metrics.

Requirements:
    - python-terrier
    - ir-measures

Install with:
    pip install python-terrier ir-measures
"""

import pandas as pd
import pyterrier as pt

# Initialize PyTerrier
if not pt.started():
    pt.init()

from ir_measures import *


def evaluate_rankings(rankings_file, dataset_name='irds:msmarco-passage-v2/trec-dl-2021/judged'):
    """
    Evaluate rankings using nDCG metrics.
    
    Args:
        rankings_file: Path to CSV file with columns [qid, docno, score]
        dataset_name: PyTerrier dataset identifier for qrels
    
    Returns:
        Dictionary of nDCG metrics
    """
    print("="*60)
    print("IPSsimRF Ranking Evaluation")
    print("="*60)
    
    # Load rankings
    print(f"\nLoading rankings from: {rankings_file}")
    df = pd.read_csv(rankings_file)
    
    print(f"Rankings loaded: {len(df)} documents across {df['qid'].nunique()} queries")
    print(f"Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['qid', 'docno', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load qrels (ground truth relevance judgments)
    print(f"\nLoading qrels from dataset: {dataset_name}")
    dataset = pt.get_dataset(dataset_name)
    qrels = dataset.get_qrels()
    
    print(f"Qrels loaded: {len(qrels)} judgments")
    
    # Compute nDCG metrics
    print("\nComputing nDCG metrics...")
    metrics = pt.Evaluate(
        df[['qid', 'docno', 'score']], 
        qrels, 
        metrics=[nDCG@1, nDCG@3, nDCG@5, nDCG@10, nDCG@30, nDCG@50, nDCG@100]
    )
    
    # Print results
    print("\n" + "="*60)
    print("nDCG Evaluation Results")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate IPSsimRF rankings with nDCG metrics"
    )
    
    parser.add_argument(
        "rankings_file",
        type=str,
        help="Path to rankings CSV file (with columns: qid, docno, score)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="irds:msmarco-passage-v2/trec-dl-2021/judged",
        help="PyTerrier dataset identifier for qrels (default: TREC DL 2021)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_rankings(args.rankings_file, args.dataset)
