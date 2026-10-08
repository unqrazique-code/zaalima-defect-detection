"""
load_test.py — Week 4 Day 4-5: API Load Testing
Zaalima Development · Project 2 · Computer Vision

Runs concurrent requests against the defect detection API
and reports throughput, latency percentiles, and success rate.

Usage:
    python src/load_test.py
    python src/load_test.py --url http://localhost:8000 --requests 200
"""

import asyncio
import io
import json
import statistics
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

try:
    import aiohttp
except ImportError:
    raise ImportError("pip install aiohttp")

BASE_URL    = "http://localhost:8000"
N_REQUESTS  = 100
CONCURRENCY = 20


def make_dummy_image_bytes() -> bytes:
    """Create a small dummy JPEG for load testing."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


async def single_request(session: aiohttp.ClientSession,
                          semaphore: asyncio.Semaphore,
                          img_bytes: bytes) -> dict:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            data = aiohttp.FormData()
            data.add_field("file", io.BytesIO(img_bytes),
                           filename="test.jpg", content_type="image/jpeg")
            async with session.post(
                f"{BASE_URL}/detect", data=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                await resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000
                return {"success": resp.status == 200, "latency_ms": latency_ms}
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {"success": False, "latency_ms": latency_ms, "error": str(e)}


async def run_load_test(base_url: str = BASE_URL,
                        n_requests: int = N_REQUESTS,
                        concurrency: int = CONCURRENCY):
    print(f"🔥 Load Test — {n_requests} requests, concurrency={concurrency}")
    print(f"   Target: POST {base_url}/detect\n")

    img_bytes = make_dummy_image_bytes()
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    t_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [single_request(session, semaphore, img_bytes)
                 for _ in range(n_requests)]
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    latencies  = [r["latency_ms"] for r in results]
    successes  = sum(1 for r in results if r["success"])
    throughput = n_requests / total_time

    lats = sorted(latencies)
    p50  = statistics.median(lats)
    p95  = lats[int(0.95 * len(lats))]
    p99  = lats[int(0.99 * len(lats))]

    print("=" * 45)
    print("LOAD TEST RESULTS")
    print("=" * 45)
    print(f"Total requests : {n_requests}")
    print(f"Successes      : {successes} ({successes/n_requests*100:.1f}%)")
    print(f"Failures       : {n_requests - successes}")
    print(f"Total time     : {total_time:.3f}s")
    print(f"Throughput     : {throughput:.1f} req/s")
    print(f"Latency p50    : {p50:.1f} ms")
    print(f"Latency p95    : {p95:.1f} ms")
    print(f"Latency p99    : {p99:.1f} ms")
    print(f"Latency min    : {min(latencies):.1f} ms")
    print(f"Latency max    : {max(latencies):.1f} ms")
    print("=" * 45)

    stats = {
        "total_requests": n_requests,
        "successes": successes,
        "success_rate": round(successes / n_requests * 100, 2),
        "total_time_s": round(total_time, 3),
        "throughput_rps": round(throughput, 1),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
    }
    Path("data/processed/load_test_results.json").write_text(json.dumps(stats, indent=2))
    print("✅ Results saved to data/processed/load_test_results.json")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",         default=BASE_URL)
    parser.add_argument("--requests",    default=N_REQUESTS,  type=int)
    parser.add_argument("--concurrency", default=CONCURRENCY, type=int)
    args = parser.parse_args()

    BASE_URL = args.url
    asyncio.run(run_load_test(args.url, args.requests, args.concurrency))
