import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for PlayListAI."""
    
    # Data processing
    symbols: str = "TRACKS_ONLY"
    session_gap_min: int = 45
    max_seq_len: int = 50
    vocab_min_freq: int = 5
    
    # API configuration
    api_port: int = 8088
    redis_url: str = "redis://localhost:6379/0"
    use_faiss: bool = False
    prom_path: str = "/metrics"
    
    # Model configuration
    model_name: str = "transformer"
    model_d: int = 256
    loss_label_smooth: float = 0.1
    
    # Performance tuning
    cost_bps: int = 3
    cand_topk: int = 500
    
    # Training
    batch_size: int = 128
    learning_rate: float = 3e-4
    num_epochs: int = 10
    dropout: float = 0.1
    weight_decay: float = 1e-4
    
    # Ray configuration
    ray_num_workers: int = 2
    ray_use_gpu: bool = False
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls(
            symbols=os.getenv('SYMBOLS', cls.symbols),
            session_gap_min=int(os.getenv('SESSION_GAP_MIN', cls.session_gap_min)),
            max_seq_len=int(os.getenv('MAX_SEQ_LEN', cls.max_seq_len)),
            vocab_min_freq=int(os.getenv('VOCAB_MIN_FREQ', cls.vocab_min_freq)),
            api_port=int(os.getenv('API_PORT', cls.api_port)),
            redis_url=os.getenv('REDIS_URL', cls.redis_url),
            use_faiss=os.getenv('USE_FAISS', 'false').lower() == 'true',
            prom_path=os.getenv('PROM_PATH', cls.prom_path),
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            model_d=int(os.getenv('MODEL_D', cls.model_d)),
            loss_label_smooth=float(os.getenv('LOSS_LABEL_SMOOTH', cls.loss_label_smooth)),
            cost_bps=int(os.getenv('COST_BPS', cls.cost_bps)),
            cand_topk=int(os.getenv('CAND_TOPK', cls.cand_topk)),
            batch_size=int(os.getenv('BATCH_SIZE', cls.batch_size)),
            learning_rate=float(os.getenv('LEARNING_RATE', cls.learning_rate)),
            num_epochs=int(os.getenv('NUM_EPOCHS', cls.num_epochs)),
            dropout=float(os.getenv('DROPOUT', cls.dropout)),
            weight_decay=float(os.getenv('WEIGHT_DECAY', cls.weight_decay)),
            ray_num_workers=int(os.getenv('RAY_NUM_WORKERS', cls.ray_num_workers)),
            ray_use_gpu=os.getenv('RAY_USE_GPU', 'false').lower() == 'true'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'symbols': self.symbols,
            'session_gap_min': self.session_gap_min,
            'max_seq_len': self.max_seq_len,
            'vocab_min_freq': self.vocab_min_freq,
            'api_port': self.api_port,
            'redis_url': self.redis_url,
            'use_faiss': self.use_faiss,
            'prom_path': self.prom_path,
            'model_name': self.model_name,
            'model_d': self.model_d,
            'loss_label_smooth': self.loss_label_smooth,
            'cost_bps': self.cost_bps,
            'cand_topk': self.cand_topk,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'ray_num_workers': self.ray_num_workers,
            'ray_use_gpu': self.ray_use_gpu
        }


# Global config instance
config = Config.from_env()

