import os
from typing import Optional
from .rank import get_ranker
from .cache import get_cache


def get_ranker():
    """Dependency to get the global ranker instance."""
    return get_ranker()


def get_cache():
    """Dependency to get the global cache instance."""
    return get_cache()


def load_environment():
    """Load environment variables and validate configuration."""
    required_vars = [
        'API_PORT',
        'MODEL_NAME',
        'MODEL_D'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return {
        'api_port': int(os.getenv('API_PORT', '8088')),
        'model_name': os.getenv('MODEL_NAME', 'transformer'),
        'model_d': int(os.getenv('MODEL_D', '256')),
        'use_faiss': os.getenv('USE_FAISS', 'false').lower() == 'true',
        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'prom_path': os.getenv('PROM_PATH', '/metrics')
    }

