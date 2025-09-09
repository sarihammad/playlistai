import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer encoder block with causal attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerModel(nn.Module):
    """SASRec-style Transformer for sequential recommendation."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = None, dropout: float = 0.1, 
                 max_seq_len: int = 50, ctx_dim: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Item embeddings
        self.item_embedding = nn.Embedding(vocab_size, d_model)
        
        # Context embeddings
        self.ctx_embedding = nn.Linear(ctx_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
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
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
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
        device = input_ids.device
        
        # Item embeddings
        item_emb = self.item_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add context embeddings if provided
        if ctx_features is not None:
            ctx_emb = self.ctx_embedding(ctx_features)  # [batch_size, seq_len, d_model]
            item_emb = item_emb + ctx_emb
        
        # Add positional encoding
        item_emb = item_emb.transpose(0, 1)  # [seq_len, batch_size, d_model]
        item_emb = self.pos_encoding(item_emb)
        item_emb = item_emb.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply dropout
        x = self.dropout(item_emb)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Output projection (tied with input embeddings)
        logits = torch.matmul(x, self.item_embedding.weight.T)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embedding matrix."""
        return self.item_embedding.weight

