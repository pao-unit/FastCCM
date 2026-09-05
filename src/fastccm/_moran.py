# _moran.py
import math

import torch

from .utils.runtime import (
    auto_batch_size_simplex,
    batch_starts,
    soft_clear,
)

class _MoranMixin:
    """Moran's I over a k-nearest-neighbour graph.

    Shares the neighbour search and the sampling with CCM, but is a
    different statistic and no CCM path calls into it."""

    def moran_matrix(
            self,
            X_emb,
            Y_emb=None,
            library_size=None,
            sample_size=None,
            exclusion_window=0,
            tp=0,
            seed=None,
            nbrs_num=None,
            sparse="auto",
            batch_size="auto",
            clean_after=True,
    ):
        """
        Compute Moran's I for target embeddings over source simplex-neighbor graphs.

        For each source embedding in `X_emb`, this builds a square graph on one
        sampled node set using the same indices for the simplex library and
        sample. The graph weights are the usual simplex exponential k-NN weights.
        Each target vector in `Y_emb` is then evaluated directly on that graph:

            I = (n / W.sum()) * (y_c.T @ W @ y_c) / (y_c.T @ y_c)

        Parameters
        ----------
        X_emb : list[array-like]
            Source embeddings, one per series, each with shape (T_x, E_x).
        Y_emb : list[array-like] or None, default None
            Target embeddings. If None, uses `X_emb`. If a target has multiple
            columns, Moran's I is computed independently for each column.
        library_size : int or {"auto", None}, default None
            Number of nodes in each square graph. Uses the same defaults as
            `score_matrix` over the valid `tp` window: None uses all valid
            points, and "auto" uses min(valid_points // 2, 700).
        sample_size : int or {"auto", None}, default None
            Alias for `library_size` for callers mirroring `score_matrix`.
            Because the Moran graph is square, both sizes must match when both
            are supplied.
        exclusion_window : int or None, default 0
            Temporal exclusion radius passed through to the simplex neighbor
            search. The default excludes self-neighbors.
        tp : int, default 0
            Optional target shift: graph nodes are built from X[t], while target
            values are drawn from Y[t + tp].
        seed : int or None, default None
            Seed for deterministic node sampling.
        nbrs_num : int or list[int], optional
            Number of simplex neighbors per source series. Defaults to E_x + 1.
        sparse : {bool, "auto"}, default "auto"
            Whether to build each graph as a sparse COO tensor. "auto" uses a
            dense graph for modest node counts and sparse storage for larger
            graphs.
        batch_size : int or {"auto", None}, default "auto"
            Number of graph rows processed per neighbor-search chunk.
        clean_after : bool, default True
            If True, run a light memory cleanup after returning.

        Returns
        -------
        dict[str, np.ndarray]
            Arrays with shape (E_y, n_Y, n_X): "I", "z", "p", "expected",
            and "var". The p-value is the upper-tail normal approximation
            `0.5 * erfc(z / sqrt(2))` using the standard Moran randomization
            variance with S0, S1, S2, and b2.
        """
        if Y_emb is None:
            Y_emb = X_emb
        size_spec = library_size
        if sample_size is not None:
            if library_size is None:
                size_spec = sample_size
            elif sample_size != library_size:
                raise ValueError(
                    "moran_matrix uses one square node set; sample_size must "
                    "match library_size or be None."
                )

        self.logger.info(
            "moran_matrix started (n_x=%d, n_y=%d, tp=%d, sparse=%s)",
            len(X_emb), len(Y_emb), int(tp), str(sparse)
        )

        result = self._moran_core(
            X_emb,
            Y_emb,
            library_size=size_spec,
            exclusion_window=exclusion_window,
            tp=tp,
            seed=seed,
            nbrs_num=nbrs_num,
            sparse=sparse,
            batch_size=batch_size,
        )

        result_np = {key: value.to("cpu").numpy() for key, value in result.items()}
        self.logger.info("moran_matrix completed with output shape %s", result_np["I"].shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return result_np

    @torch.inference_mode()
    def _moran_core(
        self,
        X_emb,
        Y_emb,
        *,
        library_size=None,
        exclusion_window=0,
        tp=0,
        seed=None,
        nbrs_num=None,
        sparse="auto",
        batch_size="auto",
    ):
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_emb)
        num_ts_Y = len(Y_emb)
        if num_ts_X == 0 or num_ts_Y == 0:
            raise ValueError("X_emb and Y_emb must be non-empty.")

        x_dims = tuple(int(X_emb[i].shape[-1]) for i in range(num_ts_X))
        y_dims = tuple(int(Y_emb[i].shape[-1]) for i in range(num_ts_Y))
        max_E_X = max(x_dims)
        max_E_Y = max(y_dims)
        min_len = min(
            min(y.shape[0] for y in Y_emb),
            min(x.shape[0] for x in X_emb)
        )
        tp = int(tp)
        if tp >= 0:
            valid_points = int(min_len - tp)
            x_offset = 0
            y_offset = tp
        else:
            valid_points = int(min_len + tp)
            x_offset = -tp
            y_offset = 0
        if valid_points <= 0:
            raise ValueError("Not enough points after applying tp.")

        # ---------- 2) graph size / node indices ----------
        library_size_mode = "explicit"
        if library_size is None:
            graph_size_res = int(valid_points)
            library_size_mode = "none->valid_points"
        elif library_size == "auto":
            graph_size_res = int(min(valid_points // 2, 700))
            library_size_mode = "auto"
        else:
            graph_size_res = int(library_size)
        graph_size_res = min(graph_size_res, int(valid_points))
        if graph_size_res <= 0:
            raise ValueError("library_size must resolve to a positive graph size.")

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        base_indices = self._get_random_indices(valid_points, graph_size_res, gen)
        graph_size = int(base_indices.numel())
        if graph_size < 4:
            raise ValueError("Moran randomization variance requires at least 4 sampled nodes.")

        x_indices = base_indices + int(x_offset)
        y_indices = base_indices + int(y_offset)
        self.logger.info(
            "Moran graph_size=%d (%s, input=%s) common_len=%d valid_points=%d tp=%d",
            int(graph_size), library_size_mode, str(library_size),
            int(min_len), int(valid_points), int(tp)
        )

        # ---------- 3) simplex neighbor params ----------
        if nbrs_num is None:
            nbrs_num = torch.tensor(
                [dim + 1 for dim in x_dims],
                device=self.device,
                dtype=torch.long,
            )
        elif isinstance(nbrs_num, int):
            nbrs_num = torch.tensor(
                [nbrs_num] * num_ts_X,
                device=self.device,
                dtype=torch.long,
            )
        else:
            if len(nbrs_num) != num_ts_X:
                raise ValueError("nbrs_num must be an int or have one entry per source series.")
            nbrs_num = torch.tensor(nbrs_num, device=self.device, dtype=torch.long)
        if (nbrs_num <= 0).any():
            raise ValueError("nbrs_num values must be positive.")
        nbrs_num_max = int(nbrs_num.max().item())
        if exclusion_window is not None and nbrs_num_max >= graph_size:
            raise ValueError(
                "nbrs_num must be smaller than graph_size when temporal "
                "exclusion is enabled."
            )
        if nbrs_num_max > graph_size:
            raise ValueError(
                "nbrs_num cannot exceed the number of sampled graph nodes. "
                f"Got nbrs_num max {nbrs_num_max} and graph_size {graph_size}."
            )

        # ---------- 4) sampling ----------
        X_nodes = self._get_random_sample(X_emb, min_len, x_indices, num_ts_X, max_E_X)
        Y_nodes = self._get_random_sample(Y_emb, min_len, y_indices, num_ts_Y, max_E_Y)

        sample_batch_size = self._resolve_moran_batch_size(
            X_nodes,
            graph_size=graph_size,
            nbrs_num_max=nbrs_num_max,
            batch_size=batch_size,
        )
        use_sparse = self._resolve_moran_sparse(sparse, graph_size)
        self.logger.info(
            "Moran neighbor graph storage=%s batch_size=%d num_batches=%d nbrs_max=%d exclusion=%s",
            "sparse" if use_sparse else "dense",
            int(sample_batch_size),
            max(1, int(math.ceil(graph_size / max(sample_batch_size, 1)))),
            int(nbrs_num_max),
            str(exclusion_window),
        )

        # ---------- 5) neighbors ----------
        neighbor_indices = torch.empty(
            (num_ts_X, graph_size, nbrs_num_max),
            device=self.device,
            dtype=torch.long,
        )
        neighbor_weights = torch.empty(
            (num_ts_X, graph_size, nbrs_num_max),
            device=self.device,
            dtype=self.dtype,
        )
        lib_index = self._prepare_nbr_library(X_nodes)
        for s0 in batch_starts(self.logger, graph_size, sample_batch_size, "moran graph batches"):
            s1 = min(graph_size, s0 + sample_batch_size)
            weights, indices = self._get_nbrs_indices_with_weights(
                X_nodes,
                X_nodes[:, s0:s1, :],
                nbrs_num,
                nbrs_num_max,
                x_indices,
                x_indices[s0:s1],
                exclusion_window,
                lib_index=lib_index,
            )
            neighbor_indices[:, s0:s1, :] = indices
            neighbor_weights[:, s0:s1, :] = weights

        # ---------- 6) Moran statistics ----------
        stat_dtype = self._moran_compute_dtype()
        out_shape = (int(max_E_Y), int(num_ts_Y), int(num_ts_X))
        result = {
            "I": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "z": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "p": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "expected": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "var": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
        }
        Y_matrix = Y_nodes.to(dtype=stat_dtype).permute(1, 0, 2).reshape(
            graph_size, num_ts_Y * max_E_Y
        ).contiguous()

        for x_idx in range(num_ts_X):
            W = self._build_moran_weight_matrix(
                neighbor_indices[x_idx],
                neighbor_weights[x_idx],
                dtype=stat_dtype,
                sparse=use_sparse,
            )
            stats = self._moran_statistics_from_W(W, Y_matrix, sparse=use_sparse)
            for key, values in stats.items():
                result[key][:, :, x_idx] = values.reshape(num_ts_Y, max_E_Y).permute(1, 0)
            del W

        valid_target_dims = torch.zeros(
            (max_E_Y, num_ts_Y),
            device=self.device,
            dtype=torch.bool,
        )
        for y_idx, dim in enumerate(y_dims):
            valid_target_dims[:dim, y_idx] = True
        invalid = ~valid_target_dims[:, :, None]
        for key in result:
            result[key] = result[key].masked_fill(invalid, float("nan"))

        return result

    def _resolve_moran_batch_size(self, X_nodes, *, graph_size, nbrs_num_max, batch_size):
        if batch_size == "auto":
            Y_dummy = torch.empty((1, graph_size, 1), device=self.device, dtype=self.dtype)
            edge_bytes = int(
                X_nodes.shape[0] * graph_size * nbrs_num_max *
                (torch.tensor([], dtype=self.dtype).element_size() + 8)
            )
            resolved, _ = auto_batch_size_simplex(
                X_nodes,
                X_nodes,
                Y_dummy,
                nbrs_num_max,
                dtype=self.dtype,
                compute_dtype=self.compute_dtype,
                budget_gb=self.memory_budget_gb,
                target_batch_size=1,
                extra_base_bytes=edge_bytes,
            )
            return max(1, int(resolved))
        if batch_size is None:
            return int(graph_size)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive, 'auto', or None.")
        return min(int(graph_size), batch_size)

    def _resolve_moran_sparse(self, sparse, graph_size):
        if sparse == "auto":
            stat_dtype = self._moran_compute_dtype()
            dense_bytes = int(
                graph_size * graph_size *
                torch.tensor([], dtype=stat_dtype).element_size()
            )
            budget_bytes = int(self.memory_budget_gb * (1024 ** 3))
            threshold = max(16 * 1024 * 1024, budget_bytes // 4)
            return dense_bytes > threshold
        if isinstance(sparse, bool):
            return sparse
        raise ValueError("sparse must be True, False, or 'auto'.")

    def _moran_compute_dtype(self):
        if str(self.device).startswith("mps"):
            return torch.float32
        return torch.float64

    def _build_moran_weight_matrix(self, indices, weights, *, dtype, sparse):
        n = int(indices.shape[0])
        weights = weights.to(dtype=dtype)
        if sparse:
            rows = torch.arange(
                n,
                device=self.device,
                dtype=torch.long,
            ).repeat_interleave(indices.shape[1])
            cols = indices.reshape(-1).to(dtype=torch.long)
            vals = weights.reshape(-1)
            keep = torch.isfinite(vals) & (vals != 0)
            return torch.sparse_coo_tensor(
                torch.stack((rows[keep], cols[keep]), dim=0),
                vals[keep],
                size=(n, n),
                device=self.device,
                dtype=dtype,
            ).coalesce()

        W = torch.zeros((n, n), device=self.device, dtype=dtype)
        W.scatter_add_(1, indices.to(dtype=torch.long), weights)
        return W

    def _moran_statistics_from_W(self, W, Y_matrix, *, sparse):
        n = int(Y_matrix.shape[0])
        dtype = Y_matrix.dtype
        n_t = torch.tensor(float(n), device=self.device, dtype=dtype)
        y_c = Y_matrix - Y_matrix.mean(dim=0, keepdim=True)

        if sparse:
            W = W.coalesce()
            edge_idx = W.indices()
            edge_vals = W.values()
            S0 = edge_vals.sum()
            row_sum = torch.zeros(n, device=self.device, dtype=dtype)
            col_sum = torch.zeros(n, device=self.device, dtype=dtype)
            if edge_vals.numel() > 0:
                row_sum.scatter_add_(0, edge_idx[0], edge_vals)
                col_sum.scatter_add_(0, edge_idx[1], edge_vals)
            sym = (W + W.transpose(0, 1)).coalesce()
            S1 = 0.5 * sym.values().pow(2).sum()
            S2 = (row_sum + col_sum).pow(2).sum()
            Wy = torch.sparse.mm(W, y_c)
        else:
            S0 = W.sum()
            sym = W + W.transpose(0, 1)
            S1 = 0.5 * sym.pow(2).sum()
            row_sum = W.sum(dim=1)
            col_sum = W.sum(dim=0)
            S2 = (row_sum + col_sum).pow(2).sum()
            Wy = W @ y_c

        den = y_c.pow(2).sum(dim=0)
        yWy = (y_c * Wy).sum(dim=0)
        eps = torch.finfo(dtype).eps
        E_scalar = -1.0 / (n_t - 1.0)
        expected = torch.empty_like(den).fill_(E_scalar)

        valid = (S0.abs() > eps) & (den > eps)
        I = torch.full_like(den, float("nan"))
        if bool(valid.any().item()):
            I = torch.where(valid, (n_t / S0) * (yWy / den), I)

        b2 = torch.full_like(den, float("nan"))
        b2 = torch.where(den > eps, n_t * y_c.pow(4).sum(dim=0) / den.pow(2), b2)

        A = n_t * (((n_t * n_t - 3.0 * n_t + 3.0) * S1) - (n_t * S2) + (3.0 * S0 * S0))
        B = ((n_t * n_t - n_t) * S1) - (2.0 * n_t * S2) + (6.0 * S0 * S0)
        denom = (n_t - 1.0) * (n_t - 2.0) * (n_t - 3.0) * S0 * S0
        var = (A - b2 * B) / denom - (E_scalar * E_scalar)
        var = torch.where((var < 0.0) & (var > -1e-12), torch.zeros_like(var), var)
        var = torch.where(
            valid & (denom.abs() > eps) & (var >= 0.0),
            var,
            torch.full_like(var, float("nan")),
        )

        z = torch.full_like(den, float("nan"))
        z_valid = valid & (var > 0.0)
        z = torch.where(z_valid, (I - E_scalar) / torch.sqrt(var), z)
        p = torch.full_like(den, float("nan"))
        p = torch.where(z_valid, 0.5 * torch.erfc(z / math.sqrt(2.0)), p)

        return {
            "I": I,
            "z": z,
            "p": p,
            "expected": expected,
            "var": var,
        }
