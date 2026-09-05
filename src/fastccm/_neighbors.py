# _neighbors.py
import torch
import numpy as np

from .utils.runtime import (
    hard_clear,
    is_oom_error,
    time_block,
    timings_summary,
)

class _NeighborLibrary:
    """
    The library side of `torch.cdist(..., "use_mm_for_euclid_dist")`, hoisted out
    of the query-batch loop.

    That mode evaluates ||q-p|| as a single matmul of augmented operands,
    [-2q, ||q||^2, 1] . [p, 1, ||p||^2]. The right-hand operand depends only on
    the library, so it is built once per call instead of once per batch.
    """

    __slots__ = ("augmented", "num_points")

    def __init__(self, augmented, num_points):
        self.augmented = augmented
        self.num_points = int(num_points)


class _NeighborsMixin:
    """Nearest-neighbour search and the sampling that feeds it.

    Distances, temporal exclusion, top-k selection and weights, over the
    reusable workspaces, plus the random library and query draws."""

    def _get_random_indices(self, num_points, sample_len, generator=None):
        #idxs_X = torch.argsort(torch.rand(num_points, device=self.device, generator=generator))[0:sample_len]

        return torch.randperm(num_points, device=self.device, generator=generator)[:sample_len]

    def _get_random_sample(self, X, min_len, indices, dim, max_E):
        if dim > 0 and self._can_stack_sample_block(X, min_len, max_E):
            if isinstance(X, torch.Tensor):
                stacked = X[:, -min_len:, :].to(device=self.device, dtype=self.dtype, copy=False)
            elif isinstance(X, np.ndarray):
                stacked = torch.as_tensor(X[:, -min_len:, :], device=self.device, dtype=self.dtype)
            else:
                first = X[0]
                if isinstance(first, torch.Tensor):
                    stacked = torch.stack(
                        [Xi[-min_len:] for Xi in X],
                        dim=0,
                    ).to(device=self.device, dtype=self.dtype, copy=False)
                else:
                    stacked_np = np.stack(
                        [np.asarray(Xi[-min_len:]) for Xi in X],
                        axis=0,
                    )
                    stacked = torch.as_tensor(stacked_np, device=self.device, dtype=self.dtype)
            return torch.index_select(stacked, 1, indices)

        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device, dtype=self.dtype)

        for i in range(dim):
            Xi_src = X[i]
            if isinstance(Xi_src, torch.Tensor):
                Xi = Xi_src.to(device=self.device, dtype=self.dtype, copy=False)[-min_len:]
            else:
                Xi = torch.as_tensor(Xi_src[-min_len:], device=self.device, dtype=self.dtype)
            X_buf[i, :, :X[i].shape[-1]] = Xi[indices]

        return X_buf

    def _can_stack_sample_block(self, X, min_len, max_E):
        if len(X) == 0:
            return False
        if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
            if X.ndim != 3:
                return False
            return (
                int(X.shape[0]) > 0 and
                int(X.shape[1]) >= int(min_len) and
                int(X.shape[2]) == int(max_E)
            )
        first = X[0]
        first_is_tensor = isinstance(first, torch.Tensor)
        first_width = int(first.shape[-1])
        if first_width != int(max_E):
            return False
        if int(first.shape[0]) < int(min_len):
            return False

        for Xi in X[1:]:
            if isinstance(Xi, torch.Tensor) != first_is_tensor:
                return False
            if int(Xi.shape[-1]) != first_width:
                return False
            if int(Xi.shape[0]) < int(min_len):
                return False
        return True

    def _use_workspace_neighbors(self):
        """
        Whether to run the neighbor search through reusable buffers instead of
        letting `cdist`/`topk` allocate per batch.

        CPU avoids first-touch page faults on large result blocks; CUDA avoids
        rebuilding the library operand and reuses intermediate allocations across
        query batches. MPS retains the native path because its `out=` support
        differs from CPU and CUDA.
        """
        return self.device.startswith(("cpu", "cuda"))

    def _release_nbr_workspace(self):
        self._nbr_workspace = {}

    def _workspace(self, name, shape, dtype):
        """Persistent flat buffer for `name`, viewed as `shape`."""
        numel = 1
        for dim in shape:
            numel *= int(dim)
        buf = self._nbr_workspace.get(name)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            self._nbr_workspace.pop(name, None)
            try:
                buf = torch.empty(numel, device=self.device, dtype=dtype)
            except RuntimeError as e:
                if is_oom_error(e):
                    self._nbr_workspace = {}
                    hard_clear(self.logger, self.device)
                raise
            self._nbr_workspace[name] = buf
        return buf[:numel].view(*shape)

    def _prepare_nbr_library(self, lib):
        """Build the library-side augmented operand once per call."""
        if not self._use_workspace_neighbors():
            return None
        lib_c = self._to_tensor(lib, dtype=self._promoted_compute_dtype())
        sq = lib_c.pow(2).sum(-1, True)
        pad = torch.ones_like(sq)
        return _NeighborLibrary(torch.cat([lib_c, pad, sq], -1), lib_c.shape[1])

    def _get_nbrs_indices_with_weights(
        self, lib, sample, n_nbrs, n_nbrs_max, lib_idx, sample_idx, exclusion_rad,
        lib_index=None, library_widths=None, trial_layout=None,
    ):
        """
        Weights and library indices of each query's k nearest neighbors.

        With `library_widths` (ascending, each <= the library length) the search
        runs once against the full library and the widths are stacked onto the
        source axis, so the result is `(n_widths * n_sources, queries, k)` with
        width as the outer part of that axis. Temporal exclusion depends only on
        the library and query time indices, never on the width, so masking the
        full block once is valid for every prefix.

        Note: the returned tensors alias per-instance workspaces and stay valid
        only until the next call, which both callers respect by consuming them
        before advancing to the next query batch.
        """
        timings = {}
        try:
            with time_block(self.logger, self.device, timings, "cdist"):
                if lib_index is None and self._use_workspace_neighbors():
                    lib_index = self._prepare_nbr_library(lib)
                if lib_index is None:
                    dist = self._cdist(sample, lib)  # (num_ts_X, S_blk, L)
                else:
                    dist = self._squared_euclidean_dist(sample, lib_index)
        except RuntimeError as e:
            if is_oom_error(e):
                hard_clear(self.logger, self.device)
            raise

        with time_block(self.logger, self.device, timings, "select"):
            if exclusion_rad is not None:
                if trial_layout is None:
                    self._exclude(dist, lib_idx, sample_idx, exclusion_rad)
                else:
                    # Each trial drew its own library and queries, so its block of
                    # the source axis carries its own excluded columns.
                    per_trial = trial_layout.sources
                    for t in range(trial_layout.trials):
                        self._exclude(dist[t * per_trial:(t + 1) * per_trial],
                                       lib_idx[t], sample_idx[t], exclusion_rad)
            if library_widths is not None:
                near_dist, indices = self._sweep_topk(
                    dist, library_widths, n_nbrs_max, squared=(lib_index is not None)
                )
                # One k per source per width, matching the stacked source axis.
                n_nbrs = n_nbrs.repeat(len(library_widths))
            elif lib_index is None:
                near_dist, indices = torch.topk(dist, n_nbrs_max, largest=False, sorted=False)
            else:
                chunk = self._prefilter_chunk(n_nbrs_max, dist.shape[2])
                if chunk:
                    near_dist, indices = self._prefilter_topk(dist, n_nbrs_max, chunk)
                else:
                    sel = (dist.shape[0], dist.shape[1], n_nbrs_max)
                    near_dist = self._workspace("near_dist", sel, dist.dtype)
                    indices = self._workspace("indices", sel, torch.long)
                    torch.topk(dist, n_nbrs_max, dim=2, largest=False, sorted=False,
                               out=(near_dist, indices))
                near_dist.clamp_min_(0).sqrt_()

        with time_block(self.logger, self.device, timings, "weights"):
            weights, indices = self._weights_from_dists(near_dist, indices, n_nbrs, n_nbrs_max)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor search timings: %s", timings_summary(timings, ["cdist", "select", "weights", "total"]))
        return weights, indices

    def _exclude(self, dist, lib_idx, sample_idx, exclusion_rad):
        """Mask the temporally excluded library columns of one distance block."""
        if self._exclude_narrow(dist, lib_idx, sample_idx, exclusion_rad):
            return
        # Keep the broadcast intermediates boolean. Subtracting int64
        # indices would materialize an 8-byte (samples x library) tensor.
        allowed = (
            (lib_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
            (lib_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
        )
        dist.masked_fill_(~allowed.unsqueeze(0), float("inf"))

    # Random-access cost of the scatter path is roughly one cache line per
    # touched element, so it only beats a streaming pass over the block while
    # the window stays much narrower than the library.
    _EXCLUSION_SCATTER_RATIO = 8

    def _exclude_narrow(self, dist, lib_idx, sample_idx, exclusion_rad):
        """
        Mark the excluded library columns of each query without touching the
        rest of the distance block.

        Only the (2r+1) time steps around a query can be excluded, so the
        columns are found through a time-index -> library-column table instead
        of comparing every query against every library point. Returns False
        when the window is wide enough that a full masked_fill_ is cheaper.
        """
        rad = int(exclusion_rad)
        num_lib = int(dist.shape[2])
        width = 2 * rad + 1
        if rad < 0 or width * self._EXCLUSION_SCATTER_RATIO >= num_lib:
            return False

        num_points = int(lib_idx.max()) + 1
        # Padded on both sides so the query-centred window never needs a bounds
        # check: absent time steps stay at -1, and query indices past the end of
        # the library (prediction mode) clamp onto the guaranteed -1 tail.
        table = torch.full((num_points + 2 * rad + 1,), -1, dtype=torch.long, device=dist.device)
        table[rad:rad + num_points].scatter_(
            0, lib_idx, torch.arange(num_lib, device=dist.device)
        )
        window = torch.arange(width, device=dist.device)
        cols = table[(sample_idx[:, None] + window[None, :]).clamp_(max=table.shape[0] - 1)]
        present = cols >= 0

        # Queries whose window holds no library point must be left alone; for the
        # rest, empty slots are folded onto a column that is excluded anyway so
        # the duplicated scatter writes stay harmless.
        first = cols.masked_fill(~present, num_lib).amin(dim=1)  # (S,)
        has_any = first < num_lib
        cols = torch.where(present, cols, (first * has_any)[:, None])

        idx = cols.unsqueeze(0).expand(dist.shape[0], -1, -1)
        if bool(has_any.all()):
            dist.scatter_(2, idx, float("inf"))
        else:
            keep = dist.gather(2, idx)
            inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
            dist.scatter_(2, idx, torch.where(has_any[None, :, None], inf, keep))
        return True

    # `topk` over the library axis is compute-bound rather than bandwidth-bound:
    # on CUDA it runs ~9-22x slower than a plain reduction over the same block.
    # Ranking chunk minima first replaces most of that scan with `amin`.
    _PREFILTER_CHUNK = 128

    _PREFILTER_MIN_LIB = 4096

    def _prefilter_chunk(self, n_nbrs_max, num_lib):
        """
        Chunk width for the prefiltered neighbor selection, or 0 for `topk`.

        The rescan covers k of the chunks, so the saving only survives while
        those are a small share of the block -- below `_PREFILTER_MIN_LIB` the
        candidate gather costs more than the skipped scan saves. CPU keeps
        `topk`, where the two run much closer together and this loses.
        """
        if not self.device.startswith("cuda"):
            return 0
        if int(num_lib) < self._PREFILTER_MIN_LIB:
            return 0
        if int(num_lib) // self._PREFILTER_CHUNK < 2 * int(n_nbrs_max):
            return 0
        return self._PREFILTER_CHUNK

    def _prefilter_topk(self, dist, k, chunk):
        """
        Exact k smallest per row, found through chunk minima.

        Each of the k smallest values sits in a chunk whose minimum is itself
        among the k smallest chunk minima -- otherwise more than k values would
        undercut it -- so rescanning only those k chunks loses nothing. Ties
        break arbitrarily, as they do in `topk`.
        """
        n_x, n_q, num_lib = dist.shape
        n_chunks = num_lib // chunk
        head = dist[..., : n_chunks * chunk].view(n_x, n_q, n_chunks, chunk)

        candidates = head.amin(-1).topk(k, dim=2, largest=False, sorted=False).indices
        scanned = head.gather(
            2, candidates[..., None].expand(n_x, n_q, k, chunk)
        ).reshape(n_x, n_q, k * chunk)
        values, flat = scanned.topk(k, dim=2, largest=False, sorted=False)
        cols = candidates.gather(
            2, flat.div(chunk, rounding_mode="floor")
        ) * chunk + flat.remainder(chunk)

        rest = num_lib - n_chunks * chunk
        if rest:
            # Columns past the last whole chunk, merged the same way.
            tail_k = min(k, rest)
            tail_v, tail_i = dist[..., n_chunks * chunk:].topk(
                tail_k, dim=2, largest=False, sorted=False
            )
            values = torch.cat([values, tail_v], 2)
            cols = torch.cat([cols, tail_i + n_chunks * chunk], 2)
            values, pick = values.topk(k, dim=2, largest=False, sorted=False)
            cols = cols.gather(2, pick)
        return values, cols

    def _slab_topk(self, block, k):
        """k smallest of one library slab, through the prefilter when it pays."""
        chunk = self._prefilter_chunk(k, block.shape[2])
        if chunk:
            return self._prefilter_topk(block, k, chunk)
        return torch.topk(block, k, dim=2, largest=False, sorted=False)

    def _sweep_topk(self, dist, widths, k, *, squared):
        """
        k nearest within each library prefix `dist[..., :w]`, `widths` ascending,
        stacked onto the source axis as `(len(widths) * n_sources, queries, k)`.

        Prefixes are nested, so the k smallest over `w` are the k smallest of the
        previous winners together with the columns that `w` adds -- anything
        beaten by k values in a prefix stays beaten in every longer one. Each
        column is therefore scanned once for the whole sweep rather than once per
        width, leaving only a (k + delta) merge per step.

        Stacking rather than returning a per-width list lets the weights, the
        reduction and the metric run once over a wider source axis instead of
        once per width, which is where the per-width dispatch cost was going.
        Width is the outer part of the combined axis, so the caller recovers
        `(width, source)` with a reshape.

        Ties break arbitrarily, as they do in `topk`, so a merged selection can
        name a different member of a tie than a flat `topk` over the prefix.
        """
        n_x, n_q, _ = dist.shape
        n_w = len(widths)
        vals = self._workspace("sweep_vals", (n_w, n_x, n_q, k), dist.dtype)
        cols = self._workspace("sweep_cols", (n_w, n_x, n_q, k), torch.long)
        prev = 0
        for wi, w in enumerate(widths):
            new_vals, new_cols = self._slab_topk(dist[..., prev:w], min(k, w - prev))
            if wi == 0:
                vals[wi] = new_vals
                cols[wi] = new_cols
            else:
                merged, pick = torch.cat([vals[wi - 1], new_vals], dim=2).topk(
                    k, dim=2, largest=False, sorted=False
                )
                vals[wi] = merged
                cols[wi] = torch.cat([cols[wi - 1], new_cols + prev], dim=2).gather(2, pick)
            prev = w
        vals = vals.reshape(n_w * n_x, n_q, k)
        if squared:
            vals.clamp_min_(0).sqrt_()
        return vals, cols.reshape(n_w * n_x, n_q, k)

    def _squared_euclidean_dist(self, sample, lib_index):
        """
        Squared `cdist(..., "use_mm_for_euclid_dist")` in a reusable buffer.

        Rank raw squared distances and apply clamp and square root only to the
        selected neighbors. Float32 cancellation can make near-zero squared
        distances slightly negative, so this may choose a different member of a
        zero-distance tie than the native path. Square-root rounding can likewise
        create ties after selection.
        """
        comp = self._promoted_compute_dtype()
        q = self._to_tensor(sample, dtype=comp)
        sq = q.pow(2).sum(-1, True)
        pad = torch.ones_like(sq)
        augmented = torch.cat([q.mul(-2), sq, pad], -1)
        dist = self._workspace(
            "dist", (q.shape[0], q.shape[1], lib_index.num_points), comp
        )
        torch.matmul(augmented, lib_index.augmented.mT, out=dist)
        return dist

    def _weights_from_dists(self, near_dist, indices, n_nbrs, n_nbrs_max):
        timings = {}
        eps = torch.finfo(near_dist.dtype).eps
        with time_block(self.logger, self.device, timings, "exp"):
            d0 = near_dist.amin(dim=2, keepdim=True).clamp_min(eps)
            w = near_dist.div(d0).neg_().exp_()
            w.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        with time_block(self.logger, self.device, timings, "mask"):
            # Uniform k (the default, nbrs_num = E_x + 1 over equal-width sources)
            # keeps every column, so the mask is only built when it can drop one.
            if int(n_nbrs.min()) < n_nbrs_max:
                keep = (torch.arange(n_nbrs_max, device=w.device).unsqueeze(0) < n_nbrs.unsqueeze(1))
                w.mul_(keep[:, None, :].to(w.dtype))

        with time_block(self.logger, self.device, timings, "normalize"):
            sumw = w.sum(dim=2, keepdim=True)
            zero = sumw <= eps
            if zero.any():
                raise RuntimeError(
                    "All neighbors excluded by `exclusion_window` for some queries. "
                    "Reduce `exclusion_window`, increase `library_size`, or ensure the "
                    "library contains valid neighbors."
                )
            out = w.div_(sumw.clamp_min(eps)).to(self.dtype)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor weight timings: %s", timings_summary(timings, ["exp", "mask", "normalize", "total"]))
        return out, indices

    def _cdist(self, a, b):
        comp = self._promoted_compute_dtype()
        try:
            a = self._to_tensor(a, dtype=comp)
            b = self._to_tensor(b, dtype=comp)
            return torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
        except RuntimeError as e:
            if is_oom_error(e):
                hard_clear(self.logger, self.device)
            raise

    def _to_tensor(self, arr, *, dtype=None, device=None):
        dtype  = self.dtype  if dtype  is None else dtype
        device = self.device if device is None else device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype, copy=False)
        return torch.as_tensor(arr, device=device, dtype=dtype)
