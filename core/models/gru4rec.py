import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRU4RecModel(nn.Module):
    """GRU4Rec model for sequential recommendation."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 50, ctx_dim: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Item embeddings
        self.item_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Context embeddings
        self.ctx_embedding = nn.Linear(ctx_dim, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, ctx_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] - input token IDs
            ctx_features: [batch_size, seq_len, ctx_dim] - context features (optional)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Item embeddings
        item_emb = self.item_embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Add context embeddings if provided
        if ctx_features is not None:
            ctx_emb = self.ctx_embedding(ctx_features)  # [batch_size, seq_len, hidden_size]
            item_emb = item_emb + ctx_emb
        
        # Apply dropout
        item_emb = self.dropout(item_emb)
        
        # GRU forward pass
        gru_out, _ = self.gru(item_emb)  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout to GRU output
        gru_out = self.dropout(gru_out)
        
        # Output projection (tied with input embeddings)
        logits = torch.matmul(gru_out, self.item_embedding.weight.T)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embedding matrix."""
        return self.item_embedding.weight

