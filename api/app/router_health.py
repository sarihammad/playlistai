from fastapi import APIRouter, Depends
from .schemas import HealthResponse
from .metrics import get_metrics
from .deps import get_ranker, get_cache
import logging

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns the current status of the service including model and cache connectivity.
    """
    try:
        ranker = get_ranker()
        cache = get_cache()
        
        model_loaded = ranker.model is not None and ranker.vocab is not None
        cache_connected = cache.is_connected()
        
        status = "ok" if model_loaded and cache_connected else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            model_loaded=model_loaded,
            cache_connected=cache_connected
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            version="1.0.0",
            model_loaded=False,
            cache_connected=False
        )


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for monitoring.
    """
    try:
        return get_metrics()
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return "Error generating metrics"

