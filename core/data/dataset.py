import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path


class PlaylistDataset(Dataset):
    """PyTorch Dataset for playlist continuation training."""
    
    def __init__(self, sequences_df: pd.DataFrame, vocab_path: str, 
                 max_seq_len: int = 50, mask_prob: float = 0.15):
        """
        Initialize dataset.
        
        Args:
            sequences_df: DataFrame with sequence data
            vocab_path: Path to vocabulary JSON file
            max_seq_len: Maximum sequence length
            mask_prob: Probability of masking tokens for training
        """
        self.sequences_df = sequences_df
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.track_to_id = vocab_data['track_to_id']
        self.id_to_track = vocab_data['id_to_track']
        self.vocab_size = vocab_data['vocab_size']
        self.pad_id = self.track_to_id['<PAD>']
        self.unk_id = self.track_to_id['<UNK>']
        self.mask_id = self.track_to_id['<MASK>']
        
        # Prepare sequences
        self.sequences = []
        for _, row in sequences_df.iterrows():
            track_seq = row['track_seq']
            ctx_seq = row['ctx_seq']
            
            # Convert tracks to IDs
            track_ids = []
            for track in track_seq:
                if track in self.track_to_id:
                    track_ids.append(self.track_to_id[track])
                else:
                    track_ids.append(self.unk_id)
            
            # Convert context to features
            ctx_features = []
            for hour, dow in ctx_seq:
                ctx_features.append([hour / 24.0, dow / 7.0])  # Normalize
            
            self.sequences.append({
                'track_ids': track_ids,
                'ctx_features': ctx_features,
                'seq_len': len(track_ids)
            })
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        seq_data = self.sequences[idx]
        track_ids = seq_data['track_ids']
        ctx_features = seq_data['ctx_features']
        seq_len = seq_data['seq_len']
        
        # Truncate or pad sequence
        if seq_len > self.max_seq_len:
            track_ids = track_ids[:self.max_seq_len]
            ctx_features = ctx_features[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Pad sequence
        while len(track_ids) < self.max_seq_len:
            track_ids.append(self.pad_id)
            ctx_features.append([0.0, 0.0])  # Default context
        
        # Create input and target sequences
        input_ids = track_ids[:-1]  # All but last
        target_ids = track_ids[1:]  # All but first
        
        # Create position IDs
        pos_ids = list(range(len(input_ids)))
        while len(pos_ids) < self.max_seq_len - 1:
            pos_ids.append(0)
        
        # Create context features for input
        input_ctx = ctx_features[:-1]
        while len(input_ctx) < self.max_seq_len - 1:
            input_ctx.append([0.0, 0.0])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * (seq_len - 1) + [0] * (self.max_seq_len - seq_len)
        
        # Apply masking for training
        if self.mask_prob > 0:
            input_ids, target_ids = self._apply_masking(input_ids, target_ids, attention_mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
            'ctx_features': torch.tensor(input_ctx, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'seq_len': torch.tensor(seq_len - 1, dtype=torch.long)  # -1 for input length
        }
    
    def _apply_masking(self, input_ids: List[int], target_ids: List[int], 
                      attention_mask: List[int]) -> Tuple[List[int], List[int]]:
        """Apply random masking to input sequence."""
        masked_input = input_ids.copy()
        masked_target = target_ids.copy()
        
        # Find positions to mask (only real tokens, not padding)
        real_positions = [i for i, mask in enumerate(attention_mask) if mask == 1]
        
        if not real_positions:
            return masked_input, masked_target
        
        # Randomly select positions to mask
        n_mask = max(1, int(len(real_positions) * self.mask_prob))
        mask_positions = np.random.choice(real_positions, size=n_mask, replace=False)
        
        for pos in mask_positions:
            # 80% of the time, replace with MASK token
            # 10% of the time, replace with random token
            # 10% of the time, keep original
            rand = np.random.random()
            if rand < 0.8:
                masked_input[pos] = self.mask_id
            elif rand < 0.9:
                masked_input[pos] = np.random.randint(3, self.vocab_size)  # Skip special tokens
        
        return masked_input, masked_target


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'pos_ids': torch.stack([item['pos_ids'] for item in batch]),
        'ctx_features': torch.stack([item['ctx_features'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'seq_len': torch.stack([item['seq_len'] for item in batch])
    }

