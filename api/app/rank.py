import torch
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
from .metrics import track_model_forward_time, track_ann_time, track_rerank_time
from .schemas import ContinueRequest, ContextRequest


class PlaylistRanker:
    """Main ranking engine for playlist continuation."""
    
    def __init__(self):
        self.model = None
        self.vocab = None
        self.id_to_track = None
        self.track_to_id = None
        self.item_embeddings = None
        self.faiss_index = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = os.getenv('MODEL_NAME', 'transformer')
        self.use_faiss = os.getenv('USE_FAISS', 'false').lower() == 'true'
        self.cand_topk = int(os.getenv('CAND_TOPK', '500'))
        
    def load_model(self):
        """Load the trained model and vocabulary."""
        try:
            # Load vocabulary
            with open('data/artifacts/vocab.json', 'r') as f:
                self.vocab = json.load(f)
            
            self.track_to_id = self.vocab['track_to_id']
            self.id_to_track = {v: k for k, v in self.track_to_id.items()}
            
            # Load model
            if self.model_name == 'transformer':
                from core.models.transformer import TransformerModel
                self.model = TransformerModel(
                    vocab_size=len(self.track_to_id),
                    d_model=int(os.getenv('MODEL_D', '256')),
                    n_heads=4,
                    n_layers=2,
                    dropout=0.1
                )
            else:  # gru4rec
                from core.models.gru4rec import GRU4RecModel
                self.model = GRU4RecModel(
                    vocab_size=len(self.track_to_id),
                    hidden_size=int(os.getenv('MODEL_D', '256')),
                    num_layers=2,
                    dropout=0.1
                )
            
            # Load weights
            checkpoint = torch.load('data/artifacts/model.pt', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load item embeddings if available
            if os.path.exists('data/artifacts/item_emb.npy'):
                self.item_embeddings = np.load('data/artifacts/item_emb.npy')
                
                # Initialize FAISS if requested
                if self.use_faiss and self.item_embeddings is not None:
                    self._init_faiss()
            
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _init_faiss(self):
        """Initialize FAISS index for approximate nearest neighbor search."""
        try:
            import faiss
            
            # Create FAISS index
            dimension = self.item_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.item_embeddings)
            self.faiss_index.add(self.item_embeddings.astype('float32'))
            
        except Exception as e:
            print(f"Failed to initialize FAISS: {e}")
            self.faiss_index = None
    
    @track_model_forward_time()
    def _forward_pass(self, input_ids: torch.Tensor, context_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run model forward pass."""
        with torch.no_grad():
            if context_features is not None:
                logits = self.model(input_ids, context_features)
            else:
                logits = self.model(input_ids)
            return logits
    
    @track_ann_time()
    def _faiss_search(self, query_embedding: np.ndarray, k: int) -> List[int]:
        """Search FAISS index for top-k candidates."""
        if self.faiss_index is None:
            return []
        
        # Normalize query
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        return indices[0].tolist()
    
    @track_rerank_time()
    def _rerank_candidates(self, input_ids: torch.Tensor, candidate_ids: List[int], 
                          context_features: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """Rerank candidates using full model."""
        if not candidate_ids:
            return []
        
        # Get logits for all candidates
        with torch.no_grad():
            if context_features is not None:
                logits = self.model(input_ids, context_features)
            else:
                logits = self.model(input_ids)
            
            # Get last position logits
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # Extract scores for candidates
            candidate_scores = []
            for track_id in candidate_ids:
                if track_id in self.id_to_track:
                    score = last_logits[track_id].item()
                    candidate_scores.append({
                        'track_id': self.id_to_track[track_id],
                        'score': score
                    })
        
        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        return candidate_scores
    
    def _apply_slate_policy(self, candidates: List[Dict[str, Any]], input_tracks: List[str], k: int) -> List[Dict[str, Any]]:
        """Apply slate policy to remove duplicates and limit repetitions."""
        # Remove tracks already in input
        filtered = [c for c in candidates if c['track_id'] not in input_tracks]
        
        # TODO: Add artist-based repetition limiting if artist mapping is available
        # For now, just return top-k
        return filtered[:k]
    
    def score_next_items(self, tracks: List[str], context: Optional[ContextRequest], 
                        k: int, use_ann: bool = False) -> List[Dict[str, Any]]:
        """Score next items for playlist continuation."""
        if not tracks or not self.model or not self.vocab:
            return []
        
        # Convert tracks to IDs
        input_ids = []
        for track in tracks:
            if track in self.track_to_id:
                input_ids.append(self.track_to_id[track])
            else:
                # Handle unknown tracks
                unk_id = self.track_to_id.get('<UNK>', 0)
                input_ids.append(unk_id)
        
        if not input_ids:
            return []
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Prepare context features if available
        context_features = None
        if context:
            ctx_vec = []
            if context.hour is not None:
                ctx_vec.append(context.hour / 24.0)  # Normalize to [0, 1]
            if context.dow is not None:
                ctx_vec.append(context.dow / 7.0)  # Normalize to [0, 1]
            
            if ctx_vec:
                context_features = torch.tensor([ctx_vec], device=self.device)
        
        if use_ann and self.faiss_index is not None and self.item_embeddings is not None:
            # Use FAISS for candidate prefiltering
            # Get query embedding (last item embedding)
            last_item_id = input_ids[-1]
            if last_item_id < len(self.item_embeddings):
                query_embedding = self.item_embeddings[last_item_id]
                candidate_ids = self._faiss_search(query_embedding, self.cand_topk)
                candidates = self._rerank_candidates(input_tensor, candidate_ids, context_features)
            else:
                # Fallback to full softmax
                logits = self._forward_pass(input_tensor, context_features)
                last_logits = logits[0, -1, :]
                scores, indices = torch.topk(last_logits, k * 2)  # Get more for filtering
                candidates = [{'track_id': self.id_to_track[idx.item()], 'score': score.item()} 
                             for score, idx in zip(scores, indices)]
        else:
            # Full softmax
            logits = self._forward_pass(input_tensor, context_features)
            last_logits = logits[0, -1, :]
            scores, indices = torch.topk(last_logits, k * 2)  # Get more for filtering
            candidates = [{'track_id': self.id_to_track[idx.item()], 'score': score.item()} 
                         for score, idx in zip(scores, indices)]
        
        # Apply slate policy
        final_candidates = self._apply_slate_policy(candidates, tracks, k)
        
        return final_candidates
    
    def cache_key(self, request: ContinueRequest) -> str:
        """Generate cache key for request."""
        # Simple cache key based on last few tracks and context
        key_parts = request.tracks[-3:]  # Last 3 tracks
        if request.context:
            if request.context.hour is not None:
                key_parts.append(f"h{request.context.hour}")
            if request.context.dow is not None:
                key_parts.append(f"d{request.context.dow}")
        
        return f"continue:{':'.join(key_parts)}:{request.k}"


# Global ranker instance
_ranker = None


def get_ranker() -> PlaylistRanker:
    """Get global ranker instance."""
    global _ranker
    if _ranker is None:
        _ranker = PlaylistRanker()
        _ranker.load_model()
    return _ranker

