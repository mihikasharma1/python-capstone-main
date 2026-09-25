"""Shared helper for calling Gemini with visible progress and a bounded timeout/retry."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_executor = ThreadPoolExecutor(max_workers=4)


def call_with_progress(make_call, *, label, total_timeout=90, poll_interval=15, retries=2, backoff=5):
    """Submits make_call() exactly once per attempt. While waiting, polls the SAME future
    (never resubmits) so a slow response never turns into a duplicate in-flight request."""
    for attempt in range(retries + 1):
        print(f"  ... {label}", flush=True)
        future = _executor.submit(make_call)
        waited = 0
        while True:
            try:
                return future.result(timeout=poll_interval)
            except FutureTimeoutError:
                waited += poll_interval
                if waited >= total_timeout:
                    if attempt == retries:
                        raise TimeoutError(f"Gemini call timed out after {waited}s while {label}.")
                    print(f"  ... gave up waiting after {waited}s, abandoning and trying once more", flush=True)
                    break  # abandon this future (still leaks the thread, but only once per real retry)
                print(f"  ... still waiting ({waited}s elapsed)", flush=True)
            except Exception as e:
                if "503" in str(e) and attempt < retries:
                    print(f"  ... model reported high demand, retrying in {backoff}s", flush=True)
                    time.sleep(backoff)
                    break
                raise