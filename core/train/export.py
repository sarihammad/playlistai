import torch
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any

from core.models.transformer import TransformerModel
from core.models.gru4rec import GRU4RecModel
from core.utils.io import save_json, save_numpy


def create_model(config: Dict[str, Any], vocab_size: int) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] == 'transformer':
        return TransformerModel(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dropout=0.0,  # No dropout for export
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    elif model_config['name'] == 'gru4rec':
        return GRU4RecModel(
            vocab_size=vocab_size,
            hidden_size=model_config['d_model'],
            num_layers=model_config['n_layers'],
            dropout=0.0,  # No dropout for export
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")


def export_model(checkpoint_path: str, vocab_path: str, config_path: str, output_dir: str):
    """Export model for serving."""
    print(f"Exporting model from {checkpoint_path}")
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = create_model(config, vocab_data['vocab_size'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Export vocabulary
    vocab_export = {
        'track_to_id': vocab_data['track_to_id'],
        'id_to_track': vocab_data['id_to_track'],
        'vocab_size': vocab_data['vocab_size'],
        'track_freqs': vocab_data.get('track_freqs', {}),
        'track_to_artist': vocab_data.get('track_to_artist', {}),
        'track_to_genre': vocab_data.get('track_to_genre', {})
    }
    
    # Save vocabulary
    vocab_output_path = Path(output_dir) / 'vocab.json'
    save_json(vocab_export, vocab_output_path)
    print(f"Saved vocabulary to {vocab_output_path}")
    
    # Save model state dict
    model_output_path = Path(output_dir) / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': vocab_data['vocab_size']
    }, model_output_path)
    print(f"Saved model to {model_output_path}")
    
    # Export item embeddings for FAISS
    item_embeddings = model.get_item_embeddings().detach().cpu().numpy()
    embeddings_output_path = Path(output_dir) / 'item_emb.npy'
    save_numpy(item_embeddings, embeddings_output_path)
    print(f"Saved item embeddings to {embeddings_output_path}")
    
    # Export model metadata
    metadata = {
        'model_name': config['model']['name'],
        'vocab_size': vocab_data['vocab_size'],
        'd_model': config['model']['d_model'],
        'max_seq_len': config['model']['max_seq_len'],
        'ctx_dim': config['model']['ctx_dim'],
        'embedding_dim': item_embeddings.shape[1],
        'export_timestamp': str(pd.Timestamp.now()),
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_val_loss': checkpoint.get('val_loss', 'unknown')
    }
    
    metadata_output_path = Path(output_dir) / 'metadata.json'
    save_json(metadata, metadata_output_path)
    print(f"Saved metadata to {metadata_output_path}")
    
    print("Export completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Export trained model for serving')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json', help='Vocabulary path')
    parser.add_argument('--config', type=str, default='core/train/config.yaml', help='Config file path')
    parser.add_argument('--out', type=str, default='data/artifacts', help='Output directory')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    export_model(args.ckpt, args.vocab, args.config, args.out)


if __name__ == '__main__':
    import pandas as pd
    main()

