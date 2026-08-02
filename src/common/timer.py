import time
from contextlib import contextmanager


@contextmanager
def timer_ms():
    """Yields a dict that gets 'ms' set on exit."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000
