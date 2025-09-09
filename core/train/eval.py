import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from torch.utils.data import DataLoader

from core.data.dataset import PlaylistDataset, collate_fn
from core.models.transformer import TransformerModel
from core.models.gru4rec import GRU4RecModel
from core.utils.metrics import calculate_metrics
from core.utils.io import save_json


def create_model(config: Dict, vocab_size: int) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] == 'transformer':
        return TransformerModel(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dropout=0.0,  # No dropout for evaluation
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    elif model_config['name'] == 'gru4rec':
        return GRU4RecModel(
            vocab_size=vocab_size,
            hidden_size=model_config['d_model'],
            num_layers=model_config['n_layers'],
            dropout=0.0,  # No dropout for evaluation
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")


def generate_recommendations(model: torch.nn.Module, dataloader: DataLoader, 
                           vocab_data: Dict, k: int = 20) -> List[Dict]:
    """Generate recommendations for all sequences in the dataset."""
    model.eval()
    device = next(model.parameters()).device
    
    id_to_track = vocab_data['id_to_track']
    all_recommendations = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get logits for the last position
            logits = model(batch['input_ids'], batch['ctx_features'])
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Get top-k recommendations
            _, top_indices = torch.topk(last_logits, k, dim=-1)
            
            # Convert to track IDs
            for i in range(batch['input_ids'].size(0)):
                recommendations = []
                for idx in top_indices[i]:
                    track_id = id_to_track.get(idx.item(), '<UNK>')
                    recommendations.append(track_id)
                
                all_recommendations.append({
                    'recommendations': recommendations,
                    'input_length': batch['seq_len'][i].item()
                })
    
    return all_recommendations


def create_ground_truth(df: pd.DataFrame) -> List[List[str]]:
    """Create ground truth from test sequences."""
    ground_truth = []
    
    for _, row in df.iterrows():
        track_seq = row['track_seq']
        # Use the last track as ground truth (next item prediction)
        if len(track_seq) > 1:
            ground_truth.append([track_seq[-1]])
        else:
            ground_truth.append([])
    
    return ground_truth


def get_catalog(df: pd.DataFrame) -> Set[str]:
    """Get all unique tracks in the dataset."""
    catalog = set()
    for _, row in df.iterrows():
        catalog.update(row['track_seq'])
    return catalog


def evaluate_model(model_path: str, test_path: str, vocab_path: str, 
                  config_path: str, output_path: str):
    """Evaluate model on test set."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Load test data
    test_df = pd.read_parquet(test_path)
    
    # Create model
    model = create_model(config, vocab_data['vocab_size'])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataset and dataloader
    test_dataset = PlaylistDataset(
        test_df, vocab_path, 
        max_seq_len=config['model']['max_seq_len'],
        mask_prob=0.0  # No masking for evaluation
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(model, test_loader, vocab_data, k=20)
    
    # Create ground truth
    ground_truth = create_ground_truth(test_df)
    
    # Get catalog
    catalog = get_catalog(test_df)
    
    # Calculate metrics
    print("Calculating metrics...")
    all_metrics = []
    
    for rec, gt in zip(recommendations, ground_truth):
        if gt:  # Only evaluate if ground truth exists
            metrics = calculate_metrics(rec['recommendations'], gt, catalog, [5, 10, 20])
            all_metrics.append(metrics)
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Add summary statistics
        avg_metrics['num_evaluations'] = len(all_metrics)
        avg_metrics['total_sequences'] = len(test_df)
        avg_metrics['catalog_size'] = len(catalog)
        
        # Save results
        save_json(avg_metrics, output_path)
        
        print("Evaluation Results:")
        print(f"  Recall@5:  {avg_metrics['recall@5']:.4f}")
        print(f"  Recall@20: {avg_metrics['recall@20']:.4f}")
        print(f"  nDCG@10:   {avg_metrics['ndcg@10']:.4f}")
        print(f"  Coverage@100: {avg_metrics['coverage@100']:.4f}")
        print(f"  Repetition@20: {avg_metrics['repetition_rate@20']:.4f}")
        print(f"  Evaluated {avg_metrics['num_evaluations']} sequences")
    else:
        print("No valid evaluations performed")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PlayListAI model')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--test', type=str, default='data/processed/test.parquet', help='Test data path')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json', help='Vocabulary path')
    parser.add_argument('--config', type=str, default='core/train/config.yaml', help='Config file path')
    parser.add_argument('--out', type=str, default='data/reports/eval.json', help='Output path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    evaluate_model(args.ckpt, args.test, args.vocab, args.config, args.out)


if __name__ == '__main__':
    main()

