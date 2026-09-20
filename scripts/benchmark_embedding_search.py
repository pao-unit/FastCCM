#!/usr/bin/env python3
"""Compare scalar, threaded, and paired embedding searches on CPU or CUDA.

The scalar and threaded baselines reproduce the original score_matrix-based
search. Paired cases call the public find_optimal_embedding_params API with
2D input. Every case uses identical trial seeds and the same common time
window. CUDA timings include transfers and synchronize the device. Each
thread worker owns a separate Functions instance.

Example:
    python scripts/benchmark_embedding_search.py --device cuda --series 64 \
        --chunks 4 8 16 32 64 --workers 2 4 --output results.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastccm import Functions
from fastccm.utils import get_td_embedding_np


def legacy_search(f, x, *, E_range, tau_range, tp_range, trials=3, seed=42,
                  **kwargs):
    """Original scalar implementation, independent of the optimized API."""
    tp_max = int(max(tp_range))
    sources = [get_td_embedding_np(x[:-tp_max, None], e, tau)[:, :, 0]
               for tau in tau_range for e in E_range]
    targets = [x[:len(x) - tp_max + tp, None] for tp in tp_range]
    result = np.mean([
        f.ccm.score_matrix(sources, targets, seed=seed + trial, **kwargs)[0].reshape(
            len(tp_range), len(tau_range), len(E_range))
        for trial in range(trials)
    ], axis=0)
    best = np.unravel_index(np.argmax(result.mean(axis=0)), result.shape[1:])
    return dict(E_range=E_range, tau_range=tau_range, tp_range=tp_range, result=result,
                optimal_tau=tau_range[best[0]], optimal_E=E_range[best[1]],
                values=result[:, best[0], best[1]])


def paired_search(f, x, *, series_chunk="auto", **kwargs):
    return np.stack([r["result"] for r in f.find_optimal_embedding_params(
        x, series_batch_size=series_chunk, **kwargs)])


class ThreadedSearch:
    def __init__(self, workers, x, kwargs, device, memory_budget_gb):
        # Construct instances serially; avoid concurrent logger initialization.
        pending = queue.SimpleQueue()
        for _ in range(workers):
            pending.put(Functions(device=device, memory_budget_gb=memory_budget_gb))
        self.local = threading.local()
        self.x, self.kwargs = x, kwargs

        def initialize():
            self.local.f = pending.get()

        self.pool = ThreadPoolExecutor(max_workers=workers, initializer=initialize)
        barrier = threading.Barrier(workers)
        list(self.pool.map(lambda _: barrier.wait(), range(workers)))

    def one(self, i):
        return legacy_search(self.local.f, self.x[:, i], **self.kwargs)["result"]

    def __call__(self):
        return np.stack(list(self.pool.map(self.one, range(self.x.shape[1]))))

    def close(self):
        self.pool.shutdown()


def compare(actual, reference):
    best = lambda a: np.argmax(a.mean(axis=1).reshape(a.shape[0], -1), axis=1)
    return dict(
        scores_match=bool(np.allclose(actual, reference, atol=1e-6, rtol=1e-5, equal_nan=True)),
        max_abs_error=float(np.nanmax(np.abs(actual - reference))),
        changed_optima=int(np.sum(best(actual) != best(reference))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--series", type=int, default=16)
    parser.add_argument("--length", type=int, default=4000)
    parser.add_argument("--max-E", type=int, default=9)
    parser.add_argument("--max-tau", type=int, default=9)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--library-size", type=int, default=700)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--workers", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--signal", choices=["gaussian", "oscillatory"], default="gaussian")
    parser.add_argument("--memory-budget-gb", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    # Do not let lower precision matrix multiplication alter neighbor selection.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rng = np.random.default_rng(123)
    x = rng.standard_normal((args.length, args.series)).astype(np.float32)
    if args.signal == "oscillatory":
        t = np.arange(args.length)[:, None]
        frequency = np.linspace(0.013, 0.071, args.series)[None, :]
        x = (np.sin(t * frequency) + 0.3 * np.cos(t * frequency * 1.731) + 0.15 * x).astype(np.float32)
    kwargs = dict(E_range=np.arange(1, args.max_E + 1), tau_range=np.arange(1, args.max_tau + 1),
        tp_range=np.array([1, 5, 10]), trials=args.trials, seed=42,
        library_size=args.library_size, sample_size=args.sample_size, exclusion_window=5)
    f = Functions(device=args.device, memory_budget_gb=args.memory_budget_gb)
    cases = {"scalar_loop": lambda: np.stack([
        legacy_search(f, x[:, i], **kwargs)["result"] for i in range(args.series)])}
    pools = []
    for workers in args.workers:
        pool = ThreadedSearch(workers, x, kwargs, args.device, args.memory_budget_gb)
        pools.append(pool)
        cases[f"thread_pool_{workers}"] = pool
    cases["paired_auto"] = lambda: paired_search(f, x, **kwargs)
    for chunk in args.chunks:
        cases[f"paired_chunk_{chunk}"] = lambda chunk=chunk: paired_search(
            f, x, **kwargs, series_chunk=chunk)

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)

    settings = vars(args).copy()
    settings["output"] = str(args.output)
    settings.update(implementation="public_api", torch=torch.__version__, numpy=np.__version__,
                    cuda=torch.version.cuda, allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    tp_range=kwargs["tp_range"].tolist(), seed=42, exclusion_window=5)
    records = {}
    report = {"settings": settings, "records": records}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        reference = cases["scalar_loop"]()
        # Validation also warms every case, including its persistent buffers.
        for name, run in cases.items():
            checks = compare(run(), reference)
            synchronize()
            records[name] = dict(**checks, seconds=[])
            print("validation", name, checks, flush=True)
            save()
            if not checks["scores_match"] or checks["changed_optima"]:
                raise AssertionError(f"{name} does not match the scalar search; timings skipped.")
        order = np.random.default_rng(99)
        for _ in range(args.repeats):
            for name in order.permutation(list(cases)):
                synchronize()
                t0 = time.perf_counter()
                actual = cases[name]()
                synchronize()
                elapsed = time.perf_counter() - t0
                checks = compare(actual, reference)
                if not checks["scores_match"] or checks["changed_optima"]:
                    raise AssertionError(f"{name} failed repeated correctness validation: {checks}")
                records[name]["seconds"].append(elapsed)
        baseline = statistics.median(records["scalar_loop"]["seconds"])
        for name, row in records.items():
            row["median_seconds"] = statistics.median(row["seconds"])
            row["speedup"] = baseline / row["median_seconds"]
            print("timing", name, row["median_seconds"], row["speedup"], flush=True)
        report["status"] = "passed"
        save()
    finally:
        for pool in pools:
            pool.close()


if __name__ == "__main__":
    main()
