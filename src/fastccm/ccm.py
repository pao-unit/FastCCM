# ccm.py
import warnings
import numpy as np
import torch
from .utils.metrics import (
    _corr_accum_dtype,
    get_metric,
    get_streaming_metric_kind,
    stream_metric_state_init,
    stream_metric_state_update,
    stream_metric_state_finalize,
)
from .utils.runtime import (
    soft_clear,
    format_bytes,
    auto_batch_size_smap,
    auto_batch_size_simplex,
    batch_starts,
)
from .utils.logger import setup_logger
from ._moran import _MoranMixin
from ._neighbors import _NeighborsMixin
from ._simplex import _SimplexMixin
from ._smap import _SmapMixin
import math
import logging
#torch.set_num_threads(os.cpu_count())  
#torch.set_num_interop_threads(1)
def _as_size_list(sizes):
    """Requested library sizes as positive ints, in the caller's own order."""
    if isinstance(sizes, (torch.Tensor, np.ndarray)):
        sizes = sizes.tolist()
    try:
        out = [int(v) for v in sizes]
    except TypeError:
        raise ValueError(
            "library_sizes must be a sequence of positive ints."
        ) from None
    if not out:
        raise ValueError("library_sizes must contain at least one size.")
    if min(out) <= 0:
        raise ValueError("library_sizes must contain only positive sizes.")
    return out


def _resolve_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    if x is None:
        return None
    key = str(x).strip().lower()
    key = {
        "float": "float32", "double": "float64", "half": "float16",
        "bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64",
        "fp16": "float16", "fp32": "float32", "fp64": "float64",
    }.get(key, key)
    dt = getattr(torch, key, None)
    if isinstance(dt, torch.dtype):
        return dt
    raise ValueError(f"Unknown dtype: {x!r}")


class _TrialLayout:
    """
    How independent trials are folded onto the source axis.

    A trial redraws both its library and its query points, so it cannot share a
    distance block with another trial. Stacking trials onto the source axis
    still pays on accelerators, where a sweep of a few series is launch-bound
    rather than compute-bound: the sources of trial `t` occupy the rows
    `[t * sources, (t + 1) * sources)` of every per-source tensor.

    `lib_idx` and `smpl_idx` are (trials, ...) index tensors -- one row per
    trial -- and are what the temporal exclusion and the target gather need in
    order to keep each block pointed at its own draw.
    """

    __slots__ = ("trials", "sources", "lib_idx", "smpl_idx")

    def __init__(self, trials, sources, lib_idx, smpl_idx):
        self.trials = int(trials)
        self.sources = int(sources)
        self.lib_idx = lib_idx
        self.smpl_idx = smpl_idx

    @property
    def rows(self):
        return self.trials * self.sources



class PairwiseCCM(_MoranMixin, _SimplexMixin, _SmapMixin, _NeighborsMixin):
    """
    Pairwise Convergent Cross Mapping (CCM) in PyTorch.

    Public API
    ----------
    • score_matrix(...)   -> CCM scores for all target/source pairs.
    • predict_matrix(...) -> Predicted target embeddings for all pairs.
    • moran_matrix(...)   -> Moran's I of target vectors over source k-NN graphs.

    Notes
    -----
    • Inputs are Python lists of array-like 2D objects (NumPy arrays or Torch
      tensors), one per time series, with shape (T, E) = (time, embedding_dim).
    • When series in the input lists have different lengths, the implementation 
      **left-truncates** each series to the common minimum length and 
      **end-aligns** them. The **last time point is assumed to match at time t**, 
      and earlier samples are dropped.
    • When running on CPU with compute_dtype=float16, distances are promoted to
      float32 to maintain numerical stability.
    """

    def __init__(self,device = "cpu", dtype="float32", compute_dtype=None,
                 verbose = 0, log_file = None, memory_budget_gb=1.0):
        """
        Create a PairwiseCCM instance.

        Parameters
        ----------
        device : {"cpu", "cuda", "cuda:0", ...}, default "cpu"
            Device on which tensors will be allocated and computations performed.
            Use a specific CUDA device string to select a GPU.
        dtype : torch.dtype or str, default "float32"
            Storage dtype for internal tensors and outputs. Accepts torch dtypes or
            common strings like {"float16","float32","float64","bfloat16"} and aliases
            {"f16","f32","f64","fp16","fp32","fp64","bf16"}.
        compute_dtype : torch.dtype or str or None, default None
            Math-ops dtype used for heavy linear algebra (e.g., distances, solves).
            If None, uses the same value as `dtype`. 
        memory_budget_gb : float, default 1.0
            Memory budget (GB) used by automatic batching (`batch_size="auto"`).
            Larger values increase batch size and speed, but use more memory.
        """
        self.device = device
        self.dtype = _resolve_dtype(dtype) or torch.float32
        self.compute_dtype = _resolve_dtype(compute_dtype) or self.dtype
        self.memory_budget_gb = float(memory_budget_gb)

        # (Optional) sanity: ensure float type
        if not (self.dtype.is_floating_point and self.compute_dtype.is_floating_point):
            raise ValueError("dtype and compute_dtype must be floating dtypes.")
        if self.memory_budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive.")
        
        self.logger = setup_logger(__name__, verbose=verbose, log_file=log_file)

        # Reusable flat buffers for the k-NN search, released by `clean_after`.
        # `cdist` has no `out=`, so it allocates its (n_X, S, L) result every
        # batch; on CPU the first-touch page faults on that block cost more than
        # the matmul that fills it.
        self._nbr_workspace = {}

    def _predict_warning_threshold_bytes(self) -> int:
        budget_bytes = max(1, int(self.memory_budget_gb * (1024 ** 3)))
        return max(1, min(64 * 1024 * 1024, budget_bytes // 8))

    def _log_predict_output_allocation(self, pred_shape, pred_bytes: int) -> None:
        level = (
            logging.WARNING
            if int(pred_bytes) >= self._predict_warning_threshold_bytes()
            else logging.INFO
        )
        self.logger.log(
            level,
            "Predict final tensor allocation shape=%s dtype=%s approx=%s on device=%s (before CPU/NumPy transfer).",
            str(pred_shape),
            str(self.dtype),
            format_bytes(pred_bytes),
            str(self.device),
        )

    def compute(self, *args, **kwargs):
        """
        DEPRECATED: Use `score_matrix(...)` instead.

        This method forwards all arguments to `score_matrix(...)` and emits a
        DeprecationWarning. See `score_matrix` for full parameter and return details.
        """
        warnings.warn("PairwiseCCM.compute → PairwiseCCM.score_matrix", DeprecationWarning)
        return self.score_matrix(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """
        DEPRECATED: Use `predict_matrix(...)` instead.

        This method forwards all arguments to `predict_matrix(...)` and emits a
        DeprecationWarning. See `predict_matrix` for full parameter and return details.
        """
        warnings.warn("PairwiseCCM.predict → PairwiseCCM.predict_matrix", DeprecationWarning)
        return self.predict_matrix(*args, **kwargs)

    def score_matrix(
            self, 
            X_emb, 
            Y_emb = None, 
            library_size = None, 
            sample_size = None, 
            exclusion_window = 0, 
            tp = 0, 
            method = "simplex", 
            seed = None, 
            metric = "corr",
            subtract_global = False,
            batch_size="auto",
            clean_after=False,
            **kwargs
    ):
        """
        Compute pairwise CCM scores for all embedding pairs.
        
        Parameters
        ----------
        X_emb : list[array-like]
            Source embeddings, one per series. Each item is 2D with shape (T_x, E_x),
            where T_x is time steps and E_x is the embedding dimension for that series.
            Accepts NumPy arrays or torch.Tensors; data are copied to `device`.
        Y_emb : list[array-like] or None, default None
            Target embeddings, same structure as X_emb. If None, uses X_emb (i.e.,
            computes all-against-all within one set).
        library_size : int or {"auto", None}, default None
            Number of library points drawn from each series:
            • None  -> use the maximum common length across series.
            • "auto"-> min(max_common_len // 2, 700).
            • int   -> use the provided count (clipped internally to valid range).
            Library indices are sampled uniformly at random (reproducible with `seed`).
        sample_size : int or {"auto", None}, default None
            Number of query (evaluation) points drawn from each series:
            • None  -> use the maximum common length across series.
            • "auto"-> min(max_common_len // 6, 250).
            • int   -> use the provided count (clipped internally to valid range).
            Sample indices are drawn uniformly at random (reproducible with `seed`).
        exclusion_window : int or None, default 0
            Temporal exclusion radius (in samples). When an integer r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (including self when r>=0).
            Use None to disable temporal exclusion entirely (self-neighbor allowed).
        tp : int, optional
            Prediction horizon. Predict Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, default "simplex"
            Local model used for neighbor-based prediction at each query point:
            • "simplex": k-NN exponential weights (Sugihara's simplex method).
            `nbrs_num` may be provided in `kwargs`; default is E_x + 1 per source series.
            • "smap"   : Locally weighted linear regression with parameter `theta`
            (default 1.0) and optional `ridge` (default 0.0) in `kwargs`.
        seed : int or None, default None
            Seed for deterministic sampling of library and sample indices.
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, default "corr"
            Scoring applied to (prediction, target).
        subtract_global : bool, default False
            If True, fit a global linear model for each source/target pair using the
            sampled library points and subtract its metric score from the local CCM
            score. The returned tensor is therefore `local_score - global_score`.
        batch_size : int or {"auto", None}, default "auto"
            Number of query points processed per chunk. If "auto", a heuristic estimates
            a safe chunk size using `memory_budget_gb` from the class constructor.
            If None,
            processes all at once (may be memory heavy).
        clean_after : bool, default False
            If True, run a cleanup after returning (calls Python GC and clears
            PyTorch/CUDA caching allocators). Keep False inside loops for
            performance.

        Other Parameters
        ----------------
        nbrs_num : int or list[int], optional (simplex only)
            Number of neighbors per source series. If int, the same k is used for all.
            Default is E_x + 1 for each source series.
        target_batch_size : {int, None, "auto"}, optional (simplex only)
            Number of target series (`n_Y`) processed per simplex reduction chunk.
            `None` keeps the original all-targets-at-once path, while
            `"auto"` applies a device-aware policy: unsplit on CUDA, calibrated
            chunking on CPU.
            When this argument is omitted, the auto policy is used by default.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.
        xtwx_precompute : bool, optional (smap only)
            Whether to precompute the per-library outer-product features used to form
            SMAP normal matrices (`X^T W X`). Default is True.
        xtwy_precompute : bool, optional (smap only)
            Whether to precompute the per-library cross terms used to form SMAP
            right-hand sides (`X^T W y`). This is usually most beneficial for
            single time series or small source/target matrices. Default is False.

        Returns
        -------
        np.ndarray
            CCM scores with shape (E_y, n_Y, n_X), where:
            • E_y : maximum embedding dimension across targets,
            • n_Y : number of target series,
            • n_X : number of source series.

        Raises
        ------
        ValueError
            If `method` is invalid or there are not enough points after applying `tp`.
        RuntimeError
            If all neighbors are excluded by `exclusion_window` for some queries.
            Out-of-memory errors are surfaced after internal cache clearing.
        """
        if Y_emb is None:
            Y_emb = X_emb

        if isinstance(library_size, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(
                "library_size must be a single size; use score_matrix_sweep() to "
                "score several library sizes in one pass."
            )

        self.logger.info(
            "score_matrix started (n_x=%d, n_y=%d, method=%s, tp=%d, metric=%s)",
            len(X_emb), len(Y_emb), method, tp, metric
        )
        r_AB = self._ccm_core(
            mode="score",
            X_lib_list=X_emb,
            Y_lib_list=Y_emb,
            X_sample_list=X_emb,
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=tp,
            method=method,
            seed=seed,
            metric=metric,
            subtract_global=subtract_global,
            batch_size=batch_size,
            **kwargs
        )

        r_AB = r_AB.to("cpu").numpy()
        self.logger.info("score_matrix completed with output shape %s", r_AB.shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return r_AB

    def score_matrix_sweep(
            self,
            X_emb,
            Y_emb = None,
            library_sizes = None,
            trials = None,
            sample_size = None,
            exclusion_window = 0,
            tp = 0,
            method = "simplex",
            seed = None,
            metric = "corr",
            subtract_global = False,
            batch_size = "auto",
            clean_after = False,
            **kwargs
    ):
        """
        Score several library sizes in a single pass over the data.

        Equivalent to calling `score_matrix` once per size with the same `seed`,
        but the shared work is done once. For a fixed seed the library indices
        are `randperm(n)[:L]`, so the libraries of the different sizes are nested
        prefixes of one permutation and the query set does not depend on the size
        at all. One library is therefore drawn at the largest size, one distance
        block is computed against it, and each size reads the matching prefix.
        Neighbor selection walks the prefixes in ascending order, merging the
        previous winners with the columns each size adds, so the library is
        scanned once for the whole sweep.

        Parameters
        ----------
        X_emb : list[array-like]
            Source embeddings, one per series; see `score_matrix`.
        Y_emb : list[array-like] or None, default None
            Target embeddings. If None, uses X_emb.
        library_sizes : sequence[int]
            Library sizes to score. Order is preserved in the output; duplicates
            are allowed. Sizes past the usable window (`min_common_len - tp`) are
            clamped to it, so they repeat the full-library result rather than
            failing. The smallest size must be at least the neighbor count
            (`nbrs_num`, by default `E_x + 1`).
        sample_size : int or {"auto", None}, default None
            Number of query points, shared by every size; see `score_matrix`.
        exclusion_window, tp, method, seed, metric, subtract_global, batch_size, clean_after
            As in `score_matrix`. `exclusion_window` depends only on the library
            and query time indices, never on the size, so it is applied once.

        trials : int or None, default None
            When given, run `trials` independent repetitions in one pass and add
            a leading trial axis to the result. Trial `t` draws exactly what
            `seed + t` would have drawn as a separate call, so a vectorised run
            reproduces the equivalent loop. The repetitions are stacked onto the
            source axis rather than looped, which mainly pays on an accelerator:
            a sweep over few series is launch-bound, and this turns `trials`
            launches into one. Not supported with `subtract_global`.

        Returns
        -------
        np.ndarray
            CCM scores with shape (n_sizes, E_y, n_Y, n_X), where `n_sizes` is
            `len(library_sizes)` and the leading axis follows the order given.
            With `trials`, the shape gains a leading trial axis:
            (trials, n_sizes, E_y, n_Y, n_X).

        Raises
        ------
        ValueError
            If `library_sizes` is empty or holds non-positive values, or if the
            smallest size is below the neighbor count.

        Notes
        -----
        Selection ties break arbitrarily, as they do in `torch.topk`, so a swept
        score can differ from the matching `score_matrix` call in the last bits
        when a query has equidistant library points.

        `method="smap"` has no swept kernel and falls back to one `score_matrix`
        call per size, which returns the same values with no shared work.
        """
        if Y_emb is None:
            Y_emb = X_emb

        sizes = _as_size_list(library_sizes)

        self.logger.info(
            "score_matrix_sweep started (n_x=%d, n_y=%d, n_sizes=%d, trials=%s, method=%s, tp=%d, metric=%s)",
            len(X_emb), len(Y_emb), len(sizes), str(trials), method, tp, metric
        )

        if method == "smap":
            if trials is not None:
                raise ValueError(
                    "trials are only vectorised for method='simplex'; call "
                    "score_matrix_sweep once per trial for smap."
                )
            self.logger.info("smap has no swept kernel; running one call per size")
            out = np.stack([
                self.score_matrix(
                    X_emb, Y_emb,
                    library_size=size,
                    sample_size=sample_size,
                    exclusion_window=exclusion_window,
                    tp=tp,
                    method=method,
                    seed=seed,
                    metric=metric,
                    subtract_global=subtract_global,
                    batch_size=batch_size,
                    clean_after=False,
                    **kwargs
                )
                for size in sizes
            ])
        else:
            r_AB = self._ccm_core(
                mode="score",
                X_lib_list=X_emb,
                Y_lib_list=Y_emb,
                X_sample_list=X_emb,
                library_size=sizes,
                sample_size=sample_size,
                exclusion_window=exclusion_window,
                tp=tp,
                method=method,
                seed=seed,
                metric=metric,
                subtract_global=subtract_global,
                batch_size=batch_size,
                trials=trials,
                **kwargs
            )
            out = r_AB.to("cpu").numpy()

        self.logger.info("score_matrix_sweep completed with output shape %s", out.shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return out

    def predict_matrix(
            self, 
            X_lib_emb, 
            Y_lib_emb = None, 
            X_pred_emb = None, 
            library_size = None, 
            exclusion_window = 0, 
            tp = 0, 
            method = "simplex", 
            seed = None,
            metric = "corr",
            subtract_global = False,
            batch_size="auto",
            clean_after=False,
            **kwargs
    ):
        """
        Predict target embeddings at given query points for all (target, source) pairs.

        Parameters
        ----------
        X_lib_emb : list[array-like]
            Source library embeddings, one per series. Each item is 2D with shape
            (T_lib, E_x). Accepts NumPy arrays or torch.Tensors.
        Y_lib_emb : list[array-like] or None, default None
            Target library embeddings, same structure as X_lib_emb. If None, uses X_lib_emb.
            Targets are aligned using Y[t + tp] so ensure `tp` leaves enough points.
        X_pred_emb : list[array-like] or None, default None
            Source query embeddings at which to evaluate predictions, one per series,
            shaped (T_pred, E_x). If None, uses X_lib_emb (predict-on-library).
        library_size : int or {"auto", None}, default None
            Number of library points drawn from each series:
            • None  -> use the maximum common library length across series.
            • "auto"-> min(max_lib_len // 2, 700).
            • int   -> use the provided count (clipped internally to valid range).
            Library indices are sampled uniformly at random (reproducible with `seed`).
        exclusion_window : int or None, default 0
            Temporal exclusion radius (in samples). When an integer r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (including self when r>=0).
            Use None to disable temporal exclusion entirely (self-neighbor allowed).
        tp : int, optional
            Prediction horizon. Predict Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, default "simplex"
            Local model used for neighbor-based prediction at each query point:
            • "simplex": k-NN exponential weights (Sugihara's simplex method).
            `nbrs_num` may be provided in `kwargs`; default is E_x + 1 per source series.
            • "smap"   : Locally weighted linear regression with parameter `theta`
            (default 1.0) and optional `ridge` (default 0.0) in `kwargs`.
        seed : int or None, default None
            Seed for deterministic sampling of library and sample indices.
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, default "corr"
            Scoring applied to (prediction, target).
        subtract_global : bool, default False
            If True, fit a global linear model for each source/target pair using the
            sampled library points and subtract its raw prediction from the local CCM
            prediction. The returned tensor is therefore
            `local_prediction - global_prediction`.
        batch_size : int or {"auto", None}, default "auto"
            Number of query points processed per chunk. If "auto", a heuristic estimates
            a safe chunk size using `memory_budget_gb` from the class constructor.
            If None,
            processes all at once (may be memory heavy).
        clean_after : bool, default False
            If True, run a cleanup after returning (calls Python GC and clears
            PyTorch/CUDA caching allocators). Keep False inside loops for
            performance.

        Other Parameters
        ----------------
        nbrs_num : int or list[int], optional (simplex only)
            Number of neighbors per source series. If int, the same k is used for all.
            Default is E_x + 1 for each source series.
        target_batch_size : {int, None, "auto"}, optional (simplex only)
            Number of target series (`n_Y`) processed per simplex reduction chunk.
            `None` keeps the original all-targets-at-once path, while
            `"auto"` applies a device-aware policy: unsplit on CUDA, calibrated
            chunking on CPU.
            When this argument is omitted, the auto policy is used by default.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.
        xtwx_precompute : bool, optional (smap only)
            Whether to precompute the per-library outer-product features used to form
            SMAP normal matrices (`X^T W X`). Default is True.
        xtwy_precompute : bool, optional (smap only)
            Whether to precompute the per-library cross terms used to form SMAP
            right-hand sides (`X^T W y`). This is usually most beneficial for
            single time series or small source/target matrices. Default is False.

        Returns
        -------
        np.ndarray
            Predicted target embeddings with shape (T_pred, E_y, n_Y, n_X), where:
            • T_pred : number of query points per source series,
            • E_y    : maximum embedding dimension across targets,
            • n_Y    : number of target series,
            • n_X    : number of source series.

        Raises
        ------
        ValueError
            If `method` is invalid or there are not enough points after applying `tp`.
        RuntimeError
            If all neighbors are excluded by `exclusion_window` for some queries.
            Out-of-memory errors are surfaced after internal cache clearing.
        """
        if Y_lib_emb is None:
            Y_lib_emb = X_lib_emb

        if isinstance(library_size, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(
                "library_size must be a single size; a library-size sweep is only "
                "available for scoring, through score_matrix_sweep()."
            )
        if X_pred_emb is None:
            X_pred_emb = X_lib_emb

        self.logger.info(
            "predict_matrix started (n_x=%d, n_y=%d, n_pred=%d, method=%s, tp=%d)",
            len(X_lib_emb), len(Y_lib_emb), len(X_pred_emb), method, tp
        )
        A = self._ccm_core(
            mode="predict",
            X_lib_list=X_lib_emb,
            Y_lib_list=Y_lib_emb,
            X_sample_list=X_pred_emb,
            library_size=library_size,
            sample_size=None, 
            exclusion_window=exclusion_window,
            tp=tp,
            method=method,
            seed=seed,
            metric=metric,
            subtract_global=subtract_global,
            batch_size=batch_size,
            **kwargs
        )

        A = A.to("cpu").numpy()
        self.logger.info("predict_matrix completed with output shape %s", A.shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return A








    @torch.inference_mode()
    def _ccm_core(
        self,
        mode,                      # "score" or "predict"
        X_lib_list,                # list[np.ndarray]
        Y_lib_list,                # list[np.ndarray]
        X_sample_list=None,        # list[np.ndarray] | None (required for "predict")
        library_size=None,
        sample_size=None,
        exclusion_window=0,
        tp=0,
        method="simplex",
        seed=None,
        metric="corr",
        subtract_global=False,
        batch_size=None,
        trials=None,
        **kwargs
    ):
        metric_fn = get_metric(metric)
        self.logger.debug(
            "__ccm_core(mode=%s, method=%s, library_size=%s, sample_size=%s, exclusion_window=%s, batch_size=%s, subtract_global=%s)",
            mode, method, library_size, sample_size, exclusion_window, batch_size, subtract_global
        )
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_lib_list)
        num_ts_Y = len(Y_lib_list)
        x_dims = tuple(int(X_lib_list[i].shape[-1]) for i in range(num_ts_X))

        max_E_X = torch.tensor([X_lib_list[i].shape[-1] for i in range(num_ts_X)], device=self.device).max().item()
        max_E_Y = torch.tensor([Y_lib_list[i].shape[-1] for i in range(num_ts_Y)], device=self.device).max().item()

        predict_output_bytes = 0
        if mode == "score":
            min_len = min(
                min(y.shape[0] for y in Y_lib_list),
                min(x.shape[0] for x in X_lib_list)
            )
            self.logger.info(
                "Embedding common_len=%d max_dim=%d",
                int(min_len), int(max(max_E_X, max_E_Y))
            )
            if min_len - tp <= 0:
                raise ValueError("Not enough points after applying tp.")
        else:
            if X_sample_list is None:
                raise ValueError("X_sample_list is required in 'predict' mode.")
            min_len_lib = torch.tensor(
                [Y_lib_list[i].shape[0] for i in range(num_ts_Y)] +
                [X_lib_list[i].shape[0] for i in range(num_ts_X)],
                device=self.device
            ).min().item()

            min_len_lib = min(
                min(y.shape[0] for y in Y_lib_list),
                min(x.shape[0] for x in X_lib_list)
            )
            min_len_pred = min(x.shape[0] for x in X_sample_list)
            self.logger.info(
                "Embedding common_lib_len=%d common_pred_len=%d max_dim=%d",
                int(min_len_lib), int(min_len_pred), int(max(max_E_X, max_E_Y))
            )

            if min_len_lib - tp <= 0 or min_len_pred <= 0:
                raise ValueError("Not enough points for library or prediction.")

        # ---------- 2) method params ----------
        if method == "simplex":
            if "nbrs_num" in kwargs:
                nbrs_num = kwargs["nbrs_num"]
                nbrs_num = torch.tensor([nbrs_num] * num_ts_X, device=self.device) if isinstance(nbrs_num, int) \
                        else torch.tensor(nbrs_num, device=self.device)
            else:
                nbrs_num = torch.tensor([X_lib_list[i].shape[-1] + 1 for i in range(num_ts_X)], device=self.device)
            target_batch_size_provided = "target_batch_size" in kwargs
            target_batch_size = kwargs.get("target_batch_size")
            if isinstance(target_batch_size, str):
                if target_batch_size != "auto":
                    raise ValueError("target_batch_size must be a positive int, None, or 'auto'.")
            elif target_batch_size is not None:
                target_batch_size = int(target_batch_size)
                if target_batch_size <= 0:
                    raise ValueError("target_batch_size must be positive, None, or 'auto'.")
            if not target_batch_size_provided:
                target_batch_size = "auto"
            global_ridge = 0.0
        elif method == "smap":
            ridge = kwargs.get("ridge", 0.0)
            theta = kwargs.get("theta", 1.0)
            xtwx_precompute = kwargs.get("xtwx_precompute", True)
            xtwy_precompute = kwargs.get("xtwy_precompute", False)
            if not isinstance(xtwx_precompute, bool):
                raise ValueError("xtwx_precompute must be a bool.")
            if not isinstance(xtwy_precompute, bool):
                raise ValueError("xtwy_precompute must be a bool.")
            global_ridge = float(ridge)
        else:
            raise ValueError("Invalid method. Supported methods are 'simplex' and 'smap'.")

        # ---------- 3) size resolution ----------
        sweep_sizes = None
        if isinstance(library_size, (list, tuple, np.ndarray, torch.Tensor)):
            if mode != "score":
                raise ValueError("A library-size sweep is only supported in 'score' mode.")
            if method != "simplex":
                raise ValueError(
                    "A library-size sweep is only supported for method='simplex'."
                )
            sweep_sizes = _as_size_list(library_size)

        if trials is not None:
            if sweep_sizes is None or mode != "score" or method != "simplex":
                raise ValueError(
                    "trials are only vectorised for a simplex library-size sweep."
                )
            if int(trials) < 1:
                raise ValueError("trials must be a positive int.")
            if subtract_global:
                raise ValueError(
                    "subtract_global is not supported with vectorised trials; "
                    "its baseline is fitted per library, which the stacked "
                    "source axis does not carry."
                )
            trials = int(trials)

        if mode == "score":
            # Defaults/auto computed from min_len (not min_len - tp)
            library_size_mode = "explicit"
            if sweep_sizes is not None:
                # One library is drawn at the largest size; every smaller size
                # reads a prefix of it, which is what `randperm(n)[:L]` already
                # produces for a fixed seed.
                library_size_res = max(sweep_sizes)
                library_size_mode = "sweep"
            elif library_size is None:
                library_size_res = min_len
                library_size_mode = "none->min_len"
            elif library_size == "auto":
                library_size_res = min(min_len // 2, 700)
                library_size_mode = "auto"
            else:
                library_size_res = int(library_size)

            sample_size_mode = "explicit"
            if sample_size is None:
                sample_size_res = min_len
                sample_size_mode = "none->min_len"
            elif sample_size == "auto":
                sample_size_res = min(min_len // 6, 250)
                sample_size_mode = "auto"
            else:
                sample_size_res = int(sample_size)

            self.logger.info(
                "library_size=%d (%s, input=%s) sample_size=%d (%s, input=%s) tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(sample_size_res), sample_size_mode, str(sample_size),
                int(tp)
            )
        else:  # predict
            library_size_mode = "explicit"
            if library_size is None:
                library_size_res = min_len_lib
                library_size_mode = "none->min_len_lib"
            elif library_size == "auto":
                library_size_res = min(min_len_lib // 2, 700)
                library_size_mode = "auto"
            else:
                library_size_res = int(library_size)
            self.logger.info(
                "library_size=%d (%s, input=%s) sample_size=%d (predict-uses-all-queries) tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(min_len_pred), int(min_len_lib), int(tp)
            )
            pred_shape = (int(min_len_pred), int(max_E_Y), int(num_ts_Y), int(num_ts_X))
            pred_bytes = int(
                pred_shape[0] * pred_shape[1] * pred_shape[2] * pred_shape[3]
                * torch.tensor([], dtype=self.dtype).element_size()
            )
            predict_output_bytes = pred_bytes
            self._log_predict_output_allocation(pred_shape, pred_bytes)

        # ---------- 4) indices ----------
        def _draw(offset):
            """Library and query indices for a run seeded `seed + offset`."""
            g_lib = g_smpl = None
            if seed is not None:
                base = int(seed) + int(offset)
                g_lib = torch.Generator(device=self.device).manual_seed(base)
                g_smpl = torch.Generator(device=self.device).manual_seed(base + 1)
            return g_lib, g_smpl

        gen_lib, gen_smpl = _draw(0)

        trial_layout = None
        if trials is not None:
            # Trial `t` draws exactly what `seed + t` drew when trials were a
            # loop of separate calls, so the vectorised run reproduces it.
            draws = [_draw(t) for t in range(trials)]
            lib_rows = [self._get_random_indices(min_len - tp, library_size_res, g) for g, _ in draws]
            smpl_rows = [self._get_random_indices(min_len - tp, sample_size_res, g) for _, g in draws]
            lib_indices, smpl_indices = lib_rows[0], smpl_rows[0]
            trial_layout = _TrialLayout(trials, num_ts_X,
                                        torch.stack(lib_rows), torch.stack(smpl_rows))
            # One neighbour count per stacked source row.
            nbrs_num = nbrs_num.repeat(trials)

        if mode == "score":
            # Indices are still drawn from the valid (min_len - tp) window, like your original
            if trial_layout is None:
                lib_indices  = self._get_random_indices(min_len - tp, library_size_res, gen_lib)
                smpl_indices = self._get_random_indices(min_len - tp, sample_size_res, gen_smpl)
            if sweep_sizes is not None:
                # `randperm(n)[:L]` silently truncates, so a requested size past
                # the usable window collapses onto the full library; fold those
                # duplicates into one computed width and map them back on return.
                n_lib = int(lib_indices.shape[0])
                sweep_widths = sorted({min(w, n_lib) for w in sweep_sizes})
                sweep_take = torch.tensor(
                    [sweep_widths.index(min(w, n_lib)) for w in sweep_sizes],
                    device=self.device, dtype=torch.long,
                )
                smallest = sweep_widths[0]
                if smallest < int(nbrs_num.max().item()):
                    raise ValueError(
                        f"library size {smallest} is smaller than the neighbor count "
                        f"{int(nbrs_num.max().item())}; raise the smallest library size "
                        "or lower `nbrs_num`."
                    )
                self.logger.info(
                    "library_size sweep widths=%s (requested=%s, usable_library=%d)",
                    str(sweep_widths), str(sweep_sizes), n_lib,
                )
        else:
            lib_indices  = self._get_random_indices(min_len_lib - tp, library_size_res, gen_lib)
            smpl_indices = torch.arange(min_len_pred, device=self.device)  # same as original

        # ---------- 5) sampling ----------
        Y_smp_trials = None
        if mode == "score" and trial_layout is not None:
            # Sources stack along the batch axis, one block of `num_ts_X` per
            # trial. Target libraries stack along the *library* axis instead, so
            # the flattened gather table stays two-dimensional and a trial's
            # neighbour columns only need a `t * library_size` offset.
            X_lib = torch.cat([
                self._get_random_sample(X_lib_list, min_len, row, num_ts_X, max_E_X)
                for row in trial_layout.lib_idx
            ], dim=0)
            X_sample = torch.cat([
                self._get_random_sample(X_lib_list, min_len, row, num_ts_X, max_E_X)
                for row in trial_layout.smpl_idx
            ], dim=0)
            Y_lib_s = torch.cat([
                self._get_random_sample(Y_lib_list, min_len, row + tp, num_ts_Y, max_E_Y)
                for row in trial_layout.lib_idx
            ], dim=1)
            Y_smp_trials = torch.stack([
                self._get_random_sample(Y_lib_list, min_len, row + tp, num_ts_Y, max_E_Y)
                for row in trial_layout.smpl_idx
            ])
            Y_smp_s = Y_smp_trials[0]
        elif mode == "score":
            X_lib    = self._get_random_sample(X_lib_list, min_len, lib_indices,  num_ts_X, max_E_X)
            X_sample = self._get_random_sample(X_lib_list, min_len, smpl_indices, num_ts_X, max_E_X)
            Y_lib_s  = self._get_random_sample(Y_lib_list, min_len, lib_indices + tp,  num_ts_Y, max_E_Y)
            Y_smp_s  = self._get_random_sample(Y_lib_list, min_len, smpl_indices + tp, num_ts_Y, max_E_Y)
        else:
            X_lib    = self._get_random_sample(X_lib_list,   min_len_lib,  lib_indices,      num_ts_X, max_E_X)
            X_sample = self._get_random_sample(X_sample_list,min_len_pred, smpl_indices,     num_ts_X, max_E_X)
            Y_lib_s  = self._get_random_sample(Y_lib_list,   min_len_lib,  lib_indices + tp, num_ts_Y, max_E_Y)
            Y_smp_s  = None

        # ---------- 6) method call ----------
        return_pred = (Y_smp_s is None)
        if method == "simplex":
            auto_batch = (batch_size == "auto")
            nbrs_num_max = nbrs_num.max().item()
            total_samples = int(X_sample.shape[1])
            # `_simplex_base_bytes` covers neither the streaming accumulators nor
            # the per-trial copies of the target tensors.
            dbytes = torch.tensor([], dtype=self.dtype).element_size()
            simplex_extra_base_bytes = 0
            if mode == "score":
                simplex_extra_base_bytes = int(Y_smp_s.numel() * dbytes)
            else:
                simplex_extra_base_bytes = int(predict_output_bytes)
            if mode == "score":
                n_w = len(sweep_widths) if sweep_sizes is not None else 1
                n_t = trials or 1
                stacked = int(num_ts_X) * n_w * n_t
                # Target-side width: one column when the target is shared
                # across the whole axis, one per (width, trial) when grouped.
                groups = 1 if trial_layout is None else n_w * n_t
                if get_streaming_metric_kind(metric_fn) is not None:
                    acc_bytes = torch.tensor(
                        [], dtype=_corr_accum_dtype(self.compute_dtype, self.device)
                    ).element_size()
                    # Three source-wide accumulators plus two target-side ones.
                    simplex_extra_base_bytes += int(
                        (3 * stacked + 2 * groups) * max_E_Y * num_ts_Y * acc_bytes
                    )
                if trial_layout is not None:
                    counted = int(num_ts_Y) * int(library_size_res) * int(max_E_Y)
                    cbytes = torch.tensor([], dtype=self.compute_dtype).element_size()
                    # Target library and its flattened copy now span every trial.
                    simplex_extra_base_bytes += int(
                        max(Y_lib_s.numel() - counted, 0) * (dbytes + cbytes)
                    )
                    # And the query targets are held per trial.
                    simplex_extra_base_bytes += int(
                        max(Y_smp_trials.numel() - Y_smp_s.numel(), 0) * dbytes
                    )
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    target_batch_size=target_batch_size,
                    extra_base_bytes=simplex_extra_base_bytes,
                    num_widths=(len(sweep_widths) if sweep_sizes is not None else 1),
                )
            else:
                _, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    target_batch_size=target_batch_size,
                    extra_base_bytes=simplex_extra_base_bytes,
                    num_widths=(len(sweep_widths) if sweep_sizes is not None else 1),
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            selected_batch_size = total_samples if batch_size is None else int(batch_size)
            selected_target_batch_size = int(batch_auto_meta["target_batch_size"])
            selected_peak_bytes = batch_auto_meta["estimated_peak_bytes"] if auto_batch else (
                batch_auto_meta["base_bytes"] + selected_batch_size * max(batch_auto_meta["per_sample_bytes"], 0)
            )
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s target_batch_size=%d target_split=%s base_est=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                selected_batch_size,
                max(1, int(math.ceil(total_samples / max(selected_batch_size, 1)))),
                str(selected_batch_size < total_samples),
                selected_target_batch_size,
                str(selected_target_batch_size < int(Y_lib_s.shape[0])),
                format_bytes(batch_auto_meta["base_bytes"]),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes(selected_peak_bytes),
            )
            out = self._simplex_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, nbrs_num, metric_fn=metric_fn,
                return_pred=return_pred, sample_batch_size=batch_size,
                nbrs_num_max=int(nbrs_num_max),
                target_batch_size=selected_target_batch_size,
                library_widths=(sweep_widths if sweep_sizes is not None else None),
                trial_layout=trial_layout,
                Y_sample_trials=Y_smp_trials,
            )


        else:
            auto_batch = (batch_size == "auto")
            total_samples = int(X_sample.shape[1])
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    xtwx_precompute=xtwx_precompute,
                    xtwy_precompute=xtwy_precompute,
                )
            else:
                _, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    xtwx_precompute=xtwx_precompute,
                    xtwy_precompute=xtwy_precompute,
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            selected_batch_size = total_samples if batch_size is None else int(batch_size)
            selected_peak_bytes = batch_auto_meta["estimated_peak_bytes"] if auto_batch else (
                batch_auto_meta["base_bytes"] + selected_batch_size * max(batch_auto_meta["per_sample_bytes"], 0)
            )
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s base_est=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                selected_batch_size,
                max(1, int(math.ceil(total_samples / max(selected_batch_size, 1)))),
                str(selected_batch_size < total_samples),
                format_bytes(batch_auto_meta["base_bytes"]),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes(selected_peak_bytes),
            )
            self.logger.info(
                "SMAP config theta=%s ridge=%s xtwx_precompute=%s xtwy_precompute=%s",
                str(theta),
                str(ridge),
                str(xtwx_precompute),
                str(xtwy_precompute),
            )


            out = self._smap_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, theta, metric_fn=metric_fn,
                return_pred=return_pred,
                sample_batch_size=batch_size,
                ridge=ridge,
                xtwx_precompute=xtwx_precompute,
                xtwy_precompute=xtwy_precompute,
            )


        if subtract_global:
            self.logger.info("Subtracting global linear baseline from %s output", mode)
            if sweep_sizes is not None:
                # The baseline is fitted on the library, so it moves with the
                # width; the library prefixes are the same ones the sweep used.
                global_out = torch.stack([
                    self._global_linear_output(
                        X_lib[:, :w],
                        X_sample,
                        Y_lib_s[:, :w],
                        Y_smp_s,
                        x_dims=x_dims,
                        metric_fn=metric_fn,
                        ridge=global_ridge,
                        sample_batch_size=batch_size,
                        return_pred=return_pred,
                    )
                    for w in sweep_widths
                ])
            else:
                global_out = self._global_linear_output(
                    X_lib,
                    X_sample,
                    Y_lib_s,
                    Y_smp_s,
                    x_dims=x_dims,
                    metric_fn=metric_fn,
                    ridge=global_ridge,
                    sample_batch_size=batch_size,
                    return_pred=return_pred,
                )
            out = out - global_out.to(device=out.device, dtype=out.dtype)

        if sweep_sizes is not None:
            axis = 0 if trial_layout is None else 1
            out = out.index_select(axis, sweep_take.to(out.device))

        return out







    def _global_linear_output(
        self,
        X_lib,
        X_sample,
        Y_lib_shifted,
        Y_sample_shifted,
        *,
        x_dims,
        metric_fn,
        ridge=0.0,
        sample_batch_size=None,
        return_pred=False,
    ):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        subset_size = X_lib.shape[1]
        subsample_size = X_sample.shape[1]
        max_E_Y = Y_lib_shifted.shape[2]
        compute_dtype = self._linear_compute_dtype()

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        Y_flat = Y_lib_shifted.to(compute_dtype).permute(1, 0, 2).reshape(
            subset_size, num_ts_Y * max_E_Y
        ).contiguous()
        beta_by_source = []
        for x_idx, ex in enumerate(x_dims):
            X_design = self._with_intercept(
                X_lib[x_idx, :, :ex].to(device=self.device, dtype=compute_dtype, copy=False)
            )
            beta_by_source.append(self._solve_global_linear_beta(X_design, Y_flat, ridge=ridge))

        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        out_device = self.device if return_pred else "cpu"
        out = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device=self.device, dtype=compute_dtype
            )
        else:
            out = torch.empty(
                (subsample_size, max_E_Y, num_ts_Y, num_ts_X),
                device=out_device,
                dtype=self.dtype,
            )

        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "global linear batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            batch_queries = s1 - s0
            A_blk = None
            if stream_kind is not None:
                A_blk = torch.empty(
                    (batch_queries, max_E_Y, num_ts_Y, num_ts_X),
                    device=self.device,
                    dtype=compute_dtype,
                )

            for x_idx, ex in enumerate(x_dims):
                Xq = self._with_intercept(
                    X_sample[x_idx, s0:s1, :ex].to(device=self.device, dtype=compute_dtype, copy=False)
                )
                pred_flat = Xq @ beta_by_source[x_idx]
                pred = pred_flat.view(batch_queries, num_ts_Y, max_E_Y).permute(0, 2, 1).contiguous()
                if stream_kind is not None:
                    A_blk[:, :, :, x_idx] = pred.to(device=self.device, dtype=compute_dtype)
                else:
                    out[s0:s1, :, :, x_idx] = pred.to(device=out_device, dtype=self.dtype)

            if stream_kind is not None:
                B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(
                    device=self.device, dtype=compute_dtype
                )[:, :, :, None].expand(batch_queries, max_E_Y, num_ts_Y, num_ts_X)
                stream_metric_state_update(stream_kind, stream_state, A_blk, B_blk)

        if stream_kind is not None:
            return stream_metric_state_finalize(stream_kind, stream_state)

        if return_pred:
            return out

        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(
            subsample_size, max_E_Y, num_ts_Y, num_ts_X
        ).to(device=out.device, dtype=compute_dtype)
        return metric_fn(out.to(dtype=compute_dtype), B_full)

    def _solve_global_linear_beta(self, X_design, Y_flat, ridge=0.0):
        XtX = X_design.transpose(0, 1) @ X_design
        XTy = X_design.transpose(0, 1) @ Y_flat
        eye = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        base = max(float(ridge), 0.0)
        jitter = tuple(base + extra for extra in (1e-8, 1e-6, 1e-4))
        for lam in jitter:
            try:
                return torch.linalg.solve(XtX + lam * eye, XTy)
            except RuntimeError:
                continue
        return torch.linalg.pinv(X_design) @ Y_flat

    def _with_intercept(self, X):
        ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=1)

    def _linear_compute_dtype(self):
        return self._promoted_compute_dtype()

    def _promoted_compute_dtype(self):
        """compute_dtype, with fp16 promoted on CPU where it is slow and unstable."""
        if self.device.startswith("cpu") and self.compute_dtype == torch.float16:
            return torch.float32
        return self.compute_dtype




















      


    def _debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)
