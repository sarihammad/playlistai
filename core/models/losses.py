import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, 
                        mask: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Compute masked cross-entropy loss with optional label smoothing.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model predictions
        targets: [batch_size, seq_len] - target token IDs
        mask: [batch_size, seq_len] - attention mask (1 for real tokens, 0 for padding)
        label_smoothing: label smoothing factor
    
    Returns:
        loss: scalar tensor
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross-entropy
    logits_flat = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    targets_flat = targets.view(-1)  # [batch_size * seq_len]
    mask_flat = mask.view(-1)  # [batch_size * seq_len]
    
    if label_smoothing > 0:
        # Label smoothing
        smooth = label_smoothing
        nll = -F.log_softmax(logits_flat, dim=-1).gather(-1, targets_flat.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -F.log_softmax(logits_flat, dim=-1).mean(-1)
        loss = (1 - smooth) * nll + smooth * smooth_loss
    else:
        # Standard cross-entropy
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Apply mask and compute mean
    masked_loss = loss * mask_flat.float()
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)


class MaskedCrossEntropyLoss(nn.Module):
    """Masked cross-entropy loss module."""
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, mask, self.label_smoothing)


def sampled_softmax_loss(logits: torch.Tensor, targets: torch.Tensor, 
                        mask: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    """
    Compute sampled softmax loss for large vocabularies.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model predictions
        targets: [batch_size, seq_len] - target token IDs
        mask: [batch_size, seq_len] - attention mask
        num_samples: number of negative samples
    
    Returns:
        loss: scalar tensor
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # Reshape for processing
    logits_flat = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    targets_flat = targets.view(-1)  # [batch_size * seq_len]
    mask_flat = mask.view(-1)  # [batch_size * seq_len]
    
    # Sample negative examples
    neg_samples = torch.randint(0, vocab_size, (batch_size * seq_len, num_samples), device=device)
    
    # Get positive and negative logits
    pos_logits = logits_flat.gather(-1, targets_flat.unsqueeze(-1)).squeeze(-1)
    neg_logits = logits_flat.gather(-1, neg_samples)  # [batch_size * seq_len, num_samples]
    
    # Compute loss
    pos_exp = torch.exp(pos_logits)
    neg_exp = torch.exp(neg_logits).sum(-1)
    
    loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
    
    # Apply mask
    masked_loss = loss * mask_flat.float()
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)


class SampledSoftmaxLoss(nn.Module):
    """Sampled softmax loss module."""
    
    def __init__(self, num_samples: int = 1000):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        return sampled_softmax_loss(logits, targets, mask, self.num_samples)

