from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ContextRequest(BaseModel):
    """Context information for playlist continuation."""
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    dow: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")


class ContinueRequest(BaseModel):
    """Request for playlist continuation."""
    tracks: List[str] = Field(..., min_items=1, description="List of track IDs in the playlist")
    k: int = Field(20, ge=1, le=100, description="Number of recommendations to return")
    context: Optional[ContextRequest] = Field(None, description="Optional context information")
    use_ann: Optional[bool] = Field(False, description="Whether to use approximate nearest neighbor search")


class ContinueItem(BaseModel):
    """A single recommendation item."""
    track_id: str = Field(..., description="Track ID")
    score: float = Field(..., description="Recommendation score")


class ContinueResponse(BaseModel):
    """Response containing playlist continuation recommendations."""
    items: List[ContinueItem] = Field(..., description="List of recommended tracks")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: Optional[str] = Field(None, description="Service version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    cache_connected: bool = Field(..., description="Whether cache is connected")

