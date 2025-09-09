from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Dict, Any
import time
import functools


# Request metrics
REQUEST_DURATION = Histogram(
    'request_duration_ms',
    'Request duration in milliseconds',
    ['route', 'method', 'code'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

REQUEST_COUNT = Counter(
    'request_total',
    'Total number of requests',
    ['route', 'method', 'code']
)

# Model performance metrics
MODEL_FORWARD_TIME = Histogram(
    'model_forward_ms',
    'Model forward pass time in milliseconds',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

ANN_TIME = Histogram(
    'ann_ms',
    'Approximate nearest neighbor search time in milliseconds',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

RERANK_TIME = Histogram(
    'rerank_ms',
    'Reranking time in milliseconds',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Cache metrics
CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio (0.0 to 1.0)'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses'
)

# Error metrics
ERRORS_TOTAL = Counter(
    'errors_total',
    'Total number of errors',
    ['route', 'error_type']
)

# Drift metrics
DRIFT_PSI = Gauge(
    'drift_psi',
    'Population Stability Index for drift detection'
)

DRIFT_KS_P = Gauge(
    'drift_ks_p',
    'Kolmogorov-Smirnov test p-value for drift detection'
)


def track_request_duration(route: str, method: str, status_code: int):
    """Track request duration and count."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(route=route, method=method, code=status_code).inc()
                return result
            except Exception as e:
                ERRORS_TOTAL.labels(route=route, error_type=type(e).__name__).inc()
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                REQUEST_DURATION.labels(route=route, method=method, code=status_code).observe(duration_ms)
        return wrapper
    return decorator


def track_model_forward_time():
    """Track model forward pass time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                MODEL_FORWARD_TIME.observe(duration_ms)
        return wrapper
    return decorator


def track_ann_time():
    """Track ANN search time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                ANN_TIME.observe(duration_ms)
        return wrapper
    return decorator


def track_rerank_time():
    """Track reranking time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                RERANK_TIME.observe(duration_ms)
        return wrapper
    return decorator


def update_cache_metrics(hit: bool):
    """Update cache hit/miss metrics."""
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
    
    # Update hit ratio
    total_requests = CACHE_HITS._value._value + CACHE_MISSES._value._value
    if total_requests > 0:
        CACHE_HIT_RATIO.set(CACHE_HITS._value._value / total_requests)


def get_metrics():
    """Get Prometheus metrics in text format."""
    return generate_latest()

