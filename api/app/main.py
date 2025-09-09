from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

from .router_health import router as health_router
from .router_continue import router as continue_router
from .deps import load_environment, get_ranker, get_cache
from .metrics import CACHE_HIT_RATIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting PlayListAI service...")
    
    try:
        # Load configuration
        config = load_environment()
        logger.info(f"Configuration loaded: {config}")
        
        # Initialize ranker (loads model and vocab)
        ranker = get_ranker()
        if ranker.model is None:
            logger.warning("Model failed to load - service will be degraded")
        else:
            logger.info("Model loaded successfully")
        
        # Initialize cache
        cache = get_cache()
        if cache.is_connected():
            logger.info("Cache connected successfully")
        else:
            logger.warning("Cache connection failed - caching disabled")
        
        # Warmup: run a simple forward pass
        try:
            warmup_tracks = ["track_1", "track_2", "track_3"]
            warmup_result = ranker.score_next_items(warmup_tracks, None, k=5)
            logger.info(f"Warmup completed, got {len(warmup_result)} recommendations")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Initialize metrics
        CACHE_HIT_RATIO.set(0.0)
        
        logger.info("PlayListAI service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PlayListAI service...")


# Create FastAPI app
app = FastAPI(
    title="PlayListAI",
    description="Context-aware playlist continuation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend and Grafana
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(continue_router)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "PlayListAI",
        "version": "1.0.0",
        "description": "Context-aware playlist continuation system",
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics",
            "continue": "/continue"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('API_PORT', '8088'))
    uvicorn.run(
        "api.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

