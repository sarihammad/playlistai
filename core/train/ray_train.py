import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ray
from ray import train
from ray.train import TorchTrainer, ScalingConfig
from ray.train.torch import TorchConfig
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any

from core.data.dataset import PlaylistDataset, collate_fn
from core.models.transformer import TransformerModel
from core.models.gru4rec import GRU4RecModel
from core.models.losses import MaskedCrossEntropyLoss
from core.utils.seed import set_seed
from core.utils.io import ensure_dir


def create_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] == 'transformer':
        return TransformerModel(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout'],
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    elif model_config['name'] == 'gru4rec':
        return GRU4RecModel(
            vocab_size=vocab_size,
            hidden_size=model_config['d_model'],
            num_layers=model_config['n_layers'],
            dropout=model_config['dropout'],
            max_seq_len=model_config['max_seq_len'],
            ctx_dim=model_config['ctx_dim']
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch['input_ids'], batch['ctx_features'])
        
        # Compute loss
        loss = criterion(logits, batch['target_ids'], batch['attention_mask'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model(batch['input_ids'], batch['ctx_features'])
            
            # Compute loss
            loss = criterion(logits, batch['target_ids'], batch['attention_mask'])
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return {'val_loss': avg_loss}


def train_worker(config: Dict[str, Any]):
    """Training worker function for Ray."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open(config['vocab_path'], 'r') as f:
        vocab_data = yaml.safe_load(f)
    vocab_size = vocab_data['vocab_size']
    
    # Create model
    model = create_model(config, vocab_size)
    model.to(device)
    
    # Create loss function
    criterion = MaskedCrossEntropyLoss(
        label_smoothing=config['training']['label_smoothing']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create datasets
    train_df = pd.read_parquet(config['train_path'])
    val_df = pd.read_parquet(config['val_path'])
    
    train_dataset = PlaylistDataset(
        train_df, config['vocab_path'], 
        max_seq_len=config['model']['max_seq_len']
    )
    val_dataset = PlaylistDataset(
        val_df, config['vocab_path'],
        max_seq_len=config['model']['max_seq_len'],
        mask_prob=0.0  # No masking for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        train.report(metrics)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': config
            }
            
            # Save to shared storage
            checkpoint_path = os.path.join(config['output']['model_dir'], 'model.pt')
            ensure_dir(checkpoint_path)
            torch.save(checkpoint, checkpoint_path)
            
            # Save item embeddings if requested
            if config['output']['save_embeddings']:
                embeddings = model.get_item_embeddings().detach().cpu().numpy()
                embeddings_path = os.path.join(config['output']['model_dir'], 'item_emb.npy')
                np.save(embeddings_path, embeddings)
            
            print(f"Saved best model at epoch {epoch} with val_loss {val_metrics['val_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train PlayListAI model with Ray')
    parser.add_argument('--config', type=str, default='core/train/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create scaling config
    scaling_config = ScalingConfig(
        num_workers=config['ray']['num_workers'],
        use_gpu=config['ray']['use_gpu'],
        resources_per_worker=config['ray']['resources_per_worker']
    )
    
    # Create trainer
    trainer = TorchTrainer(
        train_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=TorchConfig(backend="gloo" if not config['ray']['use_gpu'] else "nccl")
    )
    
    # Train
    print("Starting training...")
    result = trainer.fit()
    
    print("Training completed!")
    print(f"Best metrics: {result.metrics}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    main()

