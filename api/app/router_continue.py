from fastapi import APIRouter, HTTPException, Depends
from .schemas import ContinueRequest, ContinueResponse, ContinueItem
from .metrics import track_request_duration, update_cache_metrics
from .deps import get_ranker, get_cache
import logging

router = APIRouter(prefix="/continue", tags=["continue"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ContinueResponse)
@track_request_duration("continue", "POST", 200)
async def continue_playlist(request: ContinueRequest):
    """
    Continue a playlist by recommending next songs.
    
    Given a partial playlist of track IDs, returns the top-k most likely
    next songs based on the trained sequence model.
    """
    if not request.tracks:
        raise HTTPException(status_code=400, detail="tracks array is required")
    
    try:
        ranker = get_ranker()
        cache = get_cache()
        
        # Check cache first
        cached_items = cache.get_cached_continuation(request)
        cache_hit = cached_items is not None
        
        if cache_hit:
            logger.info(f"Cache hit for playlist continuation")
            items = cached_items
        else:
            logger.info(f"Cache miss, computing recommendations")
            # Generate recommendations
            items = ranker.score_next_items(
                tracks=request.tracks,
                context=request.context,
                k=request.k,
                use_ann=request.use_ann or False
            )
            
            # Cache the result
            if items:
                cache.cache_continuation(request, items)
        
        # Update cache metrics
        update_cache_metrics(cache_hit)
        
        # Convert to response format
        response_items = [ContinueItem(**item) for item in items]
        
        logger.info(f"Returning {len(response_items)} recommendations")
        return ContinueResponse(items=response_items)
        
    except Exception as e:
        logger.error(f"Error in playlist continuation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during playlist continuation")

