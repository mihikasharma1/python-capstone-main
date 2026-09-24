"""Shared helper for calling Gemini with visible progress and a bounded timeout/retry."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_executor = ThreadPoolExecutor(max_workers=4)


def call_with_progress(make_call, *, label, timeout=30, retries=1, backoff=3):
    """Run make_call() (a zero-arg lambda wrapping a generate_content call), printing a
    progress line first, and never hanging silently past `timeout` seconds."""
    print(f"  ... {label}", flush=True)
    for attempt in range(retries + 1):
        future = _executor.submit(make_call)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            if attempt == retries:
                raise TimeoutError(f"Gemini call timed out after {timeout}s while {label}.")
            print(f"  ... still waiting after {timeout}s, retrying", flush=True)
        except Exception as e:
            if "503" in str(e) and attempt < retries:
                print(f"  ... model reported high demand, retrying in {backoff}s", flush=True)
                time.sleep(backoff)
                continue
            raise