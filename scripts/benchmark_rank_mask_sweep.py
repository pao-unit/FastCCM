#!/usr/bin/env python3
"""
Evaluate rank masking as a shared-work selector for library-size sweeps.

`score_matrix_sweep` shares one distance block across library sizes because the
sizes are nested prefixes of one permutation. Trials cannot share it the same
way: each trial is a *different* permutation, so its library is a scattered
column subset, and re-indexing columns costs ~5x the GEMM it would save.

Rank masking sidesteps that. A uniform library of size w is a uniform thinning
of the point set, so its k nearest neighbours are exactly the first k survivors
of that thinning applied to the fully ranked neighbour list -- anything beaten by
k kept points stays beaten. A trial is just another draw of the same thinning.
So one distance block plus one deep ranking serves every size and every trial:
sizes become nested masks, repetitions independent masks, and no distance is
ever recomputed. The selector is exact, and `check_exact` asserts it.

Measured on CPU (50x50 sources/targets, 5000 points, 10 auto sizes), fitting
total cost as `one-time + slope * trials`:

    naive  :    8ms one-time +  92.3ms/trial   (slope: 20.1 scoring + 72.2 selection)
    prefix :   27ms one-time +  57.7ms/trial   (slope: 20.1 scoring + 37.6 selection)
    mask   :   70ms one-time +  40.0ms/trial   (slope: 20.1 scoring + 19.9 selection)

The shared work is only the one-time term -- 4% of the total at 40 trials -- so
what rank masking actually buys is a cheaper *per-trial* selection: a uint8
membership gather and a cumsum instead of a repeated `topk`. Hence it wins on
slope but needs enough trials to pay off its deeper ranking:

    trials      naive    prefix     mask   prefix/mask
         3     279.8ms   179.5ms  193.2ms     0.93x
        10     947.7ms   599.7ms  468.4ms     1.28x
        20    1901.7ms  1216.8ms  882.8ms     1.38x
    (1000 sources, 4 targets, 10 trials)       1.73x

Crossover is near 10 trials, and rises with the sweep's smallest library size
(`w_min/N`) since the required ranking depth goes as k*N/w_min. `library_sizes=
"auto"` sets `w_min/N = 0.01`, the least favourable end.

Measured on CUDA (H100 PCIe, torch 2.2.2+cu118), selector stage, 50 sources:

    trials   per-trial  rank-masked  speedup
         3      11.2ms         8.1ms    1.40x
        10      37.1ms        23.7ms    1.57x
        40     147.8ms        83.0ms    1.78x
    (1000 sources, 250 queries, 10 trials)    1.83x

So CUDA does *not* change the picture the way the `__prefilter_topk` comment
suggests it might: the gains match CPU's (~1.8x ceiling), with the crossover
moving down from ~5 trials to ~3. One CUDA-specific cost partly offsets the
rest -- the shortfall check reads a scalar back per (trial, width), forcing a
device sync that the prefix selector never pays. Batching those checks, or
budgeting depth conservatively and leaning on the fallback, would remove it.

Not in `src/`: `Functions.convergence_test` defaults to `trials=3`, where the
shipped prefix selector is better on CPU, and this path would add a second
selector, depth escalation, an exact fallback and a third tie-breaking regime to
the hottest loop -- for a selector-stage gain that end-to-end is diluted by the
scoring stage. The shipped sweep already gives 6.3-8.4x over the per-size loop
on CUDA, so the marginal headroom here is modest.

Usage:
    python scripts/benchmark_rank_mask_sweep.py [--device cuda] [--trials 10]
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

K_DEFAULT = 4
EMBED_DIM = 3
DEPTH_C = 3          # depth budget multiple of the mean requirement k*|U|/w
TOP_MULT = 2         # escalation headroom above the deepest per-size budget


def auto_sizes(n_points: int, num: int = 10) -> list[int]:
    """The spacing `Functions.convergence_test(library_sizes="auto")` produces."""
    lo = max(n_points // 100, 10)
    hi = max(n_points, lo + 1)
    return np.unique(np.logspace(np.log10(lo), np.log10(hi), num=num, dtype=int)).tolist()


def squared_dist(queries, library):
    """Same operand shape as `PairwiseCCM.__squared_euclidean_dist`."""
    q = torch.cat([queries.mul(-2), queries.pow(2).sum(-1, True),
                   torch.ones_like(queries[..., :1])], -1)
    lib = torch.cat([library, torch.ones_like(library[..., :1]),
                     library.pow(2).sum(-1, True)], -1)
    return torch.matmul(q, lib.mT)


def prefix_merge(dist, widths, k):
    """The shipped selector: k nearest within each nested prefix of one block."""
    out, vals, cols, prev = [], None, None, 0
    for w in widths:
        new_vals, new_cols = torch.topk(dist[..., prev:w], min(k, w - prev),
                                        dim=2, largest=False, sorted=False)
        if vals is None:
            vals, cols = new_vals, new_cols + prev
        else:
            vals, pick = torch.cat([vals, new_vals], 2).topk(k, dim=2, largest=False, sorted=False)
            cols = torch.cat([cols, new_cols + prev], 2).gather(2, pick)
        prev = w
        out.append((vals.clone(), cols.clone()))
    return out


def select_per_trial(queries, points, perms, sizes, k):
    """Baseline: one distance block per trial, prefix merge across sizes."""
    picks = {}
    largest = max(sizes)
    for trial, perm in enumerate(perms):
        library = perm[:largest]
        dist = squared_dist(queries, points[:, library])
        for w, (vals, cols) in zip(sizes, prefix_merge(dist, sizes, k)):
            picks[(trial, w)] = (vals, library[cols])
    return picks


def select_rank_masked(queries, points, perms, sizes, k, n_points, stats=None):
    """
    One distance block and one deep ranking, shared by every trial and size.

    Sizes below the crossover keep the prefix merge, on a single per-trial gather
    of the widest of them -- for those the library is small enough that ranking it
    directly beats ranking deep enough to find k survivors. Sizes above it read
    the shared ranking through `rank < w`.
    """
    union = torch.unique(torch.cat([perm[:max(sizes)] for perm in perms]))
    n_union = int(union.numel())
    dist = squared_dist(queries, points[:, union])                    # shared

    crossover = max(k + 1, int((DEPTH_C * k * n_union) ** 0.5))
    small = [w for w in sizes if w < crossover]
    large = [w for w in sizes if w >= crossover]
    deepest = DEPTH_C * k * n_union // max(large[0], 1) + k if large else k
    top_m = min(n_union, max(TOP_MULT * deepest, deepest + k))
    ranked_vals, ranked_cols = torch.topk(dist, top_m, dim=2, largest=False, sorted=True)

    picks = {}
    head = max(small) if small else 0
    for trial, perm in enumerate(perms):
        rank = torch.empty(n_points, dtype=torch.long, device=perm.device)
        rank[perm] = torch.arange(n_points, device=perm.device)
        rank_of_union = rank[union]                                   # once per trial
        want = torch.arange(1, k + 1, dtype=torch.int32,
                            device=dist.device).expand(*ranked_cols.shape[:2], k).contiguous()

        if small:
            head_cols = perm[:head]
            head_dist = dist[..., torch.searchsorted(union, head_cols)]
            for w, (vals, cols) in zip(small, prefix_merge(head_dist, small, k)):
                picks[(trial, w)] = (vals, head_cols[cols])

        for w in large:
            # Membership is precomputed as one byte per union point, so the hot
            # lookup reads 1 byte per ranked neighbour instead of gathering an
            # 8-byte rank and comparing it.
            member = (rank_of_union < w).to(torch.uint8)
            depth = min(top_m, DEPTH_C * k * n_union // w + k)
            while True:
                survivors = member[ranked_cols[..., :depth]].cumsum(2, dtype=torch.int32)
                if int((survivors[..., -1] < k).sum()) == 0 or depth >= top_m:
                    break
                depth = min(top_m, depth * 2)
                if stats is not None:
                    stats["deepen"] += 1
            if int((survivors[..., -1] < k).sum()):                   # ceiling: exact fallback
                cols = perm[:w]
                vals, idx = torch.topk(dist[..., torch.searchsorted(union, cols)], k,
                                       dim=2, largest=False, sorted=True)
                picks[(trial, w)] = (vals, cols[idx])
                if stats is not None:
                    stats["fallback"] += 1
                continue
            # The survivor count is monotone, so the j-th survivor sits at the
            # first position where it reaches j -- a binary search, not a sort.
            sel = torch.searchsorted(survivors.contiguous(), want)
            picks[(trial, w)] = (ranked_vals[..., :depth].gather(2, sel),
                                 union[ranked_cols[..., :depth].gather(2, sel)])
            if stats is not None:
                stats["masked"] += 1
    return picks


def make_case(n_sources, n_queries, n_points, n_trials, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.rand(n_sources, n_points, EMBED_DIM, generator=gen).to(device)
    queries = torch.rand(n_sources, n_queries, EMBED_DIM, generator=gen).to(device)
    perms = [torch.randperm(n_points, generator=torch.Generator(device="cpu").manual_seed(100 + t)).to(device)
             for t in range(n_trials)]
    return queries, points, perms


def check_exact(device, k=K_DEFAULT):
    print("exactness (neighbour sets must match the per-trial selector)")
    for n_sources, n_queries, n_points, n_trials in ((3, 96, 1200, 6), (2, 250, 3000, 4)):
        queries, points, perms = make_case(n_sources, n_queries, n_points, n_trials, device)
        sizes = auto_sizes(n_points, num=8)
        stats = {"masked": 0, "deepen": 0, "fallback": 0}
        ref = select_per_trial(queries, points, perms, sizes, k)
        got = select_rank_masked(queries, points, perms, sizes, k, n_points, stats)
        bad = 0
        for key, (ref_vals, ref_cols) in ref.items():
            got_vals, got_cols = got[key]
            same = (torch.equal(ref_cols.sort(2).values, got_cols.sort(2).values)
                    and torch.allclose(ref_vals.sort(2).values, got_vals.sort(2).values, atol=1e-4))
            bad += 0 if same else 1
        print(f"  sources={n_sources} queries={n_queries} points={n_points} trials={n_trials}: "
              f"{len(ref) - bad}/{len(ref)} pairs identical  {stats}")
        assert bad == 0, "rank masking must select the same neighbours"


def timed(fn, repeats=3, device="cpu"):
    """
    Median wall time in ms.

    CUDA launches are asynchronous, so both variants must be synchronised or the
    one that happens to read a scalar back (the mask path checks for queries
    short of k) is the only one actually waited on.
    """
    cuda = str(device).startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    fn()
    sync()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=K_DEFAULT)
    args = parser.parse_args()

    check_exact(args.device, args.neighbors)

    print(f"\nselector cost, {args.trials} trials x 10 sizes, device={args.device}")
    print(f"{'shape':<38} {'per-trial':>11} {'rank-masked':>12} {'speedup':>8}")
    shapes = ((1, 250, 5000), (20, 250, 5000), (50, 250, 5000), (1000, 250, 5000))
    for n_sources, n_queries, n_points in shapes:
        queries, points, perms = make_case(n_sources, n_queries, n_points, args.trials, args.device)
        sizes = auto_sizes(n_points)
        k = args.neighbors
        base = timed(lambda: select_per_trial(queries, points, perms, sizes, k), device=args.device)
        mask = timed(lambda: select_rank_masked(queries, points, perms, sizes, k, n_points), device=args.device)
        label = f"sources={n_sources} queries={n_queries} points={n_points}"
        print(f"{label:<38} {base:10.1f}ms {mask:11.1f}ms {base / mask:7.2f}x")

    print("\nselector cost vs trial count (sources=50, queries=250, points=5000)")
    print(f"{'trials':>7} {'per-trial':>11} {'rank-masked':>12} {'speedup':>8}")
    for n_trials in (3, 5, 10, 20, 40):
        queries, points, perms = make_case(50, 250, 5000, n_trials, args.device)
        sizes = auto_sizes(5000)
        base = timed(lambda: select_per_trial(queries, points, perms, sizes, args.neighbors), device=args.device)
        mask = timed(lambda: select_rank_masked(queries, points, perms, sizes, args.neighbors, 5000), device=args.device)
        print(f"{n_trials:7d} {base:10.1f}ms {mask:11.1f}ms {base / mask:7.2f}x")


if __name__ == "__main__":
    main()
