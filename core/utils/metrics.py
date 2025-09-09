import numpy as np
from typing import List, Set
from collections import Counter


def recall_at_k(recommendations: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K metric."""
    if not ground_truth:
        return 0.0
    
    top_k = recommendations[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth)


def ndcg_at_k(recommendations: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate nDCG@K metric."""
    if not ground_truth:
        return 0.0
    
    # Create relevance scores (1 if in ground truth, 0 otherwise)
    relevance = [1 if rec in ground_truth else 0 for rec in recommendations[:k]]
    
    # Calculate DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    
    # Calculate IDCG (ideal DCG)
    ideal_relevance = [1] * min(len(ground_truth), k)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    
    return dcg / idcg if idcg > 0 else 0.0


def coverage_at_k(recommendations: List[str], catalog: Set[str], k: int) -> float:
    """Calculate Coverage@K metric - fraction of catalog covered in top-k."""
    if not catalog:
        return 0.0
    
    top_k = recommendations[:k]
    covered = len(set(top_k) & catalog)
    return covered / len(catalog)


def repetition_rate(recommendations: List[str], k: int) -> float:
    """Calculate repetition rate in top-k recommendations."""
    if k <= 1:
        return 0.0
    
    top_k = recommendations[:k]
    unique_items = len(set(top_k))
    return 1.0 - (unique_items / k)


def diversity_metrics(recommendations: List[str], k: int) -> dict:
    """Calculate diversity metrics for recommendations."""
    top_k = recommendations[:k]
    
    # Intra-list diversity (average pairwise distance)
    # For simplicity, using Jaccard distance on item sets
    if len(top_k) < 2:
        return {'intra_list_diversity': 0.0, 'unique_items': len(set(top_k))}
    
    # Count unique items
    unique_items = len(set(top_k))
    
    # Simple diversity measure: ratio of unique items
    diversity = unique_items / len(top_k)
    
    return {
        'intra_list_diversity': diversity,
        'unique_items': unique_items,
        'total_items': len(top_k)
    }


def calculate_metrics(recommendations: List[str], ground_truth: List[str], 
                     catalog: Set[str], k_values: List[int] = [5, 10, 20]) -> dict:
    """Calculate comprehensive metrics for recommendations."""
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(recommendations, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(recommendations, ground_truth, k)
        metrics[f'coverage@{k}'] = coverage_at_k(recommendations, catalog, k)
    
    # Repetition rate for the largest k
    max_k = max(k_values)
    metrics[f'repetition_rate@{max_k}'] = repetition_rate(recommendations, max_k)
    
    # Diversity metrics
    diversity = diversity_metrics(recommendations, max_k)
    metrics.update(diversity)
    
    return metrics

