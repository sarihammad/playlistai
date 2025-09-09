import redis
import json
import hashlib
from typing import List, Dict, Any, Optional
import os
from .schemas import ContinueRequest


class CacheManager:
    """Redis-based cache manager for playlist continuations."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.client = None
        self.ttl = 300  # 5 minutes default TTL
        
    def connect(self):
        """Connect to Redis."""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            return True
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def _generate_key(self, request: ContinueRequest, model_version: str = "default") -> str:
        """Generate cache key from request."""
        # Create a hash of the request components
        key_data = {
            'model_version': model_version,
            'tracks': request.tracks,
            'k': request.k,
            'context': request.context.dict() if request.context else None,
            'use_ann': request.use_ann
        }
        
        # Create deterministic hash
        key_str = json.dumps(key_data, sort_keys=True)
        return f"playlistai:continue:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result."""
        if not self.is_connected():
            return None
        
        try:
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: List[Dict[str, Any]], ttl: int = None) -> bool:
        """Set cached result."""
        if not self.is_connected():
            return False
        
        try:
            ttl = ttl or self.ttl
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def get_cached_continuation(self, request: ContinueRequest, model_version: str = "default") -> Optional[List[Dict[str, Any]]]:
        """Get cached playlist continuation."""
        key = self._generate_key(request, model_version)
        return self.get(key)
    
    def cache_continuation(self, request: ContinueRequest, items: List[Dict[str, Any]], model_version: str = "default") -> bool:
        """Cache playlist continuation result."""
        key = self._generate_key(request, model_version)
        return self.set(key, items)


# Global cache instance
_cache_manager = None


def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        _cache_manager.connect()
    return _cache_manager

