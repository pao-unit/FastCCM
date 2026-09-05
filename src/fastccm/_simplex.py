# _simplex.py
import logging
import math

import torch

from .utils.metrics import (
    get_streaming_metric_kind,
    stream_metric_state_finalize,
    stream_metric_state_init,
    stream_metric_state_update,
)
from .utils.runtime import (
    batch_starts,
    format_bytes,
    hard_clear,
    is_oom_error,
    resolve_simplex_target_batch_size,
    tic,
    time_block,
    timings_summary,
    toc_ms,
)

class _SimplexMixin:
    """k-NN weighted average prediction, and the reduce-path choice.

    Library sizes and trials arrive stacked on the source axis, so the
    weights, the reduction and the metric each run once over a wider axis."""

    @torch.inference_mode()
    def _simplex_prediction(self, lib_indices, smpl_indices,
                              X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, 
                              exclusion_rad, nbrs_num, metric_fn, return_pred=False, sample_batch_size=None,
                              nbrs_num_max=None, target_batch_size=None, library_widths=None,
                              trial_layout=None, Y_sample_trials=None):
        num_src_ts = X_lib.shape[0]
        # Widths stack onto the source axis; the target tables are built against
        # the full library, so a width needs no re-indexing.
        sweep = library_widths is not None
        n_widths = len(library_widths) if sweep else 1
        num_ts_X = num_src_ts * n_widths
        # With trials vectorised, `num_src_ts` already covers (trial, source) and
        # each trial's neighbour columns index its own slice of the target table.
        n_trials = 1 if trial_layout is None else trial_layout.trials
        per_trial_src = num_src_ts if trial_layout is None else trial_layout.sources
        lib_span = int(Y_lib_shifted.shape[1]) // n_trials
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_Y = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]

        if (sample_batch_size is None) or (sample_batch_size >= subsample_size):
            sample_batch_size = subsample_size
        target_batch_size = resolve_simplex_target_batch_size(num_ts_Y, target_batch_size)

        if nbrs_num_max is None:
            nbrs_num_max = int(nbrs_num.max().item())

        lib_index = self._prepare_nbr_library(X_lib)
        self.logger.debug(
            "Entering simplex backend (queries=%d, library_points=%d, num_sources=%d, num_widths=%d, num_targets=%d, Ey=%d, nbrs_max=%d, batch_size=%d, num_batches=%d, target_batch_size=%d, num_target_batches=%d, exclusion=%s)",
            int(subsample_size),
            int(X_lib.shape[1]),
            int(num_src_ts),
            int(n_widths),
            int(num_ts_Y),
            int(max_E_Y),
            int(nbrs_num_max),
            int(sample_batch_size),
            int(math.ceil(subsample_size / sample_batch_size)),
            int(target_batch_size),
            int(math.ceil(num_ts_Y / target_batch_size)),
            str(exclusion_rad),
        )
        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind,
                max_E_Y,
                num_ts_Y,
                num_ts_X,
                device=self.device,
                dtype=self.compute_dtype,
                shared_target=(trial_layout is None),
                target_group=(1 if trial_layout is None else per_trial_src),
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        # Flatten target features once so per-batch gathers can use index_select on a
        # dense 2D table instead of slower advanced indexing over a 3D tensor.
        Y_lib_flat = Y_lib_shifted.to(self.compute_dtype).permute(1, 0, 2).reshape(
            Y_lib_shifted.shape[1], num_ts_Y * max_E_Y
        ).contiguous()
        # Row (width, trial, source) of the stacked axis reads trial `t`'s block
        # of that table, so its neighbour columns shift by `t * lib_span`.
        col_shift = None
        if trial_layout is not None:
            rows = torch.arange(num_ts_X, device=self.device)
            col_shift = (((rows // per_trial_src) % n_trials) * lib_span).reshape(-1, 1, 1)
            trial_of_col = (rows // per_trial_src) % n_trials
            # One target column per group of sources rather than per source:
            # the streaming metric broadcasts it across the group, so the
            # target-side reductions stay `sources` times narrower.
            trial_of_group = torch.arange(
                num_ts_X // per_trial_src, device=self.device
            ) % n_trials
        target_ranges = tuple(
            (y0, min(num_ts_Y, y0 + target_batch_size))
            for y0 in range(0, num_ts_Y, target_batch_size)
        )
        max_target_block_width = int(target_batch_size * max_E_Y)
        # Only the gather + `bmm` path needs the k-fold staging buffers.
        if self._use_fused_reduce(
            nbrs_num_max, max_target_block_width, rows=num_ts_X * sample_batch_size
        ):
            gather_buf = None
            reduce_buf = None
            bag_offsets = torch.arange(
                0, num_ts_X * sample_batch_size * nbrs_num_max, nbrs_num_max,
                device=Y_lib_flat.device, dtype=torch.long,
            )
        else:
            bag_offsets = None
            gather_buf = torch.empty(
                (num_ts_X * sample_batch_size * nbrs_num_max, max_target_block_width),
                device=Y_lib_flat.device,
                dtype=self.compute_dtype,
            )
            reduce_buf = torch.empty(
                (num_ts_X * sample_batch_size, 1, max_target_block_width),
                device=Y_lib_flat.device,
                dtype=self.compute_dtype,
            )
        #nbrs_mask = (torch.arange(nbrs_num_max).unsqueeze(0) < nbrs_num.unsqueeze(1))
        A = None if stream_kind is not None else torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=out_device, dtype=self.dtype)
        if sweep and A is not None:
            # A metric that cannot stream needs every query of a width before it
            # can score, so the stored block carries all widths at once. The
            # streaming metrics keep accumulators instead and do not pay this.
            self.logger.log(
                logging.WARNING if A.numel() * A.element_size() >= self._predict_warning_threshold_bytes()
                else logging.INFO,
                "Sweep prediction block spans %d widths: shape=%s approx=%s on %s. "
                "A streaming metric (corr, mse, rmse, mae, neg_nrmse) keeps only accumulators.",
                n_widths, str(tuple(A.shape)), format_bytes(A.numel() * A.element_size()), str(out_device),
            )
        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "simplex batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            self.logger.debug(
                "Simplex batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = tic(self.logger, self.device) if self._debug_enabled() else None

            try:
                with time_block(self.logger, self.device, timings, "neighbors"):
                    X_sample_b = X_sample[:, s0:s1, :].to(device=self.device, dtype=self.dtype, copy=False)
                    weights, indices = self._get_nbrs_indices_with_weights(
                        X_lib, X_sample_b,
                        nbrs_num, nbrs_num_max,
                        lib_indices if trial_layout is None else trial_layout.lib_idx,
                        smpl_indices[s0:s1] if trial_layout is None
                        else trial_layout.smpl_idx[:, s0:s1],
                        exclusion_rad, lib_index=lib_index,
                        library_widths=library_widths,
                        trial_layout=trial_layout,
                    )
                    if col_shift is not None:
                        indices = indices + col_shift
            except RuntimeError as e:
                if is_oom_error(e):
                    self.logger.warning("OOM in simplex batch [%d:%d): %s", int(s0), int(s1), str(e))
                    hard_clear(self.logger, self.device)
                raise

            weights_c = weights.to(self.compute_dtype)
            batch_queries = s1 - s0
            flat_indices = indices.reshape(-1)
            weights_rows = weights_c.reshape(num_ts_X * batch_queries, 1, nbrs_num_max)
            weights_flat = weights_c.reshape(-1)
            flat_count = int(flat_indices.numel())
            batch_rows = int(num_ts_X * batch_queries)
            Y_sample_batch_view = None
            if stream_kind is not None:
                if trial_layout is None:
                    Y_sample_batch_view = Y_sample_shifted[:, s0:s1, :].permute(1, 2, 0).to(
                        device=self.device, dtype=self.compute_dtype
                    ).contiguous()
                else:
                    # (trials, targets, queries, E) -> (queries, E, targets, trials)
                    Y_sample_batch_view = Y_sample_trials[:, :, s0:s1, :].permute(2, 3, 1, 0).to(
                        device=self.device, dtype=self.compute_dtype
                    ).contiguous()
            gather_ms = 0.0
            weighted_avg_ms = 0.0
            metric_ms = 0.0
            store_ms = 0.0
            for y0, y1 in target_ranges:
                y_width = (y1 - y0) * max_E_Y

                # Decide on the full batch's row count, not this batch's: the
                # staging buffers are sized for `sample_batch_size`, and a
                # partial last batch must not flip the choice and look for a
                # buffer that was never allocated.
                fused = bag_offsets is not None and self._use_fused_reduce(
                    nbrs_num_max, y_width, rows=num_ts_X * sample_batch_size
                )

                t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                Y_src = Y_lib_flat[:, y0 * max_E_Y:y1 * max_E_Y]
                if fused:
                    Y_idx = None  # folded into the weighted sum below
                elif y_width == max_target_block_width:
                    Y_idx = torch.index_select(
                        Y_src,
                        0,
                        flat_indices,
                        out=gather_buf[:flat_count],
                    ).reshape(batch_rows, nbrs_num_max, y_width)
                else:
                    Y_idx = torch.index_select(
                        Y_src,
                        0,
                        flat_indices,
                    ).reshape(batch_rows, nbrs_num_max, y_width)
                if t_part is not None:
                    gather_ms += toc_ms(self.logger, self.device, t_part)

                t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                if fused:
                    A_y = torch.nn.functional.embedding_bag(
                        flat_indices,
                        Y_src,
                        bag_offsets[:batch_rows],
                        mode="sum",
                        per_sample_weights=weights_flat,
                    ).reshape(num_ts_X, batch_queries, y1 - y0, max_E_Y)
                elif y_width == max_target_block_width:
                    A_y = torch.bmm(
                        weights_rows,
                        Y_idx,
                        out=reduce_buf[:batch_rows],
                    ).reshape(num_ts_X, batch_queries, y1 - y0, max_E_Y)
                else:
                    A_y = torch.bmm(weights_rows, Y_idx).reshape(
                        num_ts_X, batch_queries, y1 - y0, max_E_Y
                    )
                A_y = A_y.permute(1, 3, 2, 0).contiguous()  # (B, E_y, y_blk, n_X)
                if t_part is not None:
                    weighted_avg_ms += toc_ms(self.logger, self.device, t_part)

                if stream_kind is not None:
                    t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                    if trial_layout is None:
                        B_y = Y_sample_batch_view[:, :, y0:y1].unsqueeze(-1).expand(
                            batch_queries, max_E_Y, y1 - y0, num_ts_X
                        )
                    else:
                        B_y = Y_sample_batch_view[:, :, y0:y1, :].index_select(3, trial_of_group)
                    stream_metric_state_update(
                        stream_kind,
                        stream_state,
                        A_y.to(device=self.device, dtype=self.compute_dtype),
                        B_y,
                        y_start=y0,
                        count_samples=(y0 == 0),
                    )
                    if t_part is not None:
                        metric_ms += toc_ms(self.logger, self.device, t_part)
                    del B_y
                else:
                    t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                    A[s0:s1, :, y0:y1, :] = A_y.to(out_device, dtype=self.dtype)
                    if t_part is not None:
                        store_ms += toc_ms(self.logger, self.device, t_part)

                del Y_idx, A_y

            if self._debug_enabled():
                timings["gather"] = gather_ms
                timings["weighted_avg"] = weighted_avg_ms
                if stream_kind is not None:
                    timings["metric"] = metric_ms
                else:
                    timings["store"] = store_ms

            if self._debug_enabled():
                timings["total"] = toc_ms(self.logger, self.device, t_batch)
                self.logger.debug(
                    "Simplex batch [%d:%d) timings: %s",
                    int(s0),
                    int(s1),
                    timings_summary(
                        timings,
                        order=["neighbors", "gather", "weighted_avg", "metric", "store", "total"],
                    ),
                )

            del weights, indices, weights_c, flat_indices, weights_rows, weights_flat, Y_sample_batch_view

        def unstack(scores):
            """
            Split the combined source axis back out as a leading width axis.

            Only the last axis is touched: `neg_nrmse` reduces over the target
            dimension, so the leading shape is the metric's to decide.
            """
            if not sweep:
                return scores
            if trial_layout is None:
                return scores.reshape(*scores.shape[:-1], n_widths, num_src_ts).movedim(-2, 0)
            # (..., width, trial, source) -> (trial, width, ..., source)
            split = scores.reshape(*scores.shape[:-1], n_widths, n_trials, per_trial_src)
            return split.movedim(-2, 0).movedim(-2, 1)

        if stream_kind is not None:
            return unstack(stream_metric_state_finalize(stream_kind, stream_state))

        if (Y_sample_shifted is None) and return_pred:
            # Prediction output; a sweep is rejected upstream, so n_widths == 1.
            return A

        self.logger.debug("Computing simplex score metric")
        if trial_layout is None:
            B = torch.permute(Y_sample_shifted, (1, 2, 0)).to(device=A.device, dtype=self.compute_dtype)[:, :, :, None] \
                .expand(Y_sample_shifted.shape[1], max_E_Y, num_ts_Y, num_ts_X)
        else:
            B = Y_sample_trials.permute(2, 3, 1, 0).to(
                device=A.device, dtype=self.compute_dtype
            ).index_select(3, trial_of_col.to(A.device))

        r_AB = metric_fn(A.to(dtype=self.compute_dtype), B)

        if return_pred:
            return (unstack(r_AB), A)
        else:
            return unstack(r_AB)

    # `bmm` on (rows, 1, k) x (rows, k, N) leaves its packed kernel once the
    # per-row right operand stops fitting the blocking it uses -- measured at
    # k*N ~ 450 for k in 2..21 -- and past that point it is ~50x slower. Stay
    # under it with margin.
    _BMM_FAST_ELEMS = 384

    # The staging block the `bmm` path materializes is (rows, k, y_width), so it
    # also grows with the row count -- which a library-size sweep multiplies by
    # the number of widths. Once it is this far past cache, allocating and
    # streaming it costs more than `bmm`'s per-row advantage saves.
    _BMM_STAGING_MAX_BYTES = 64 * 1024 * 1024

    def _use_fused_reduce(self, nbrs_num_max, y_width, rows=None):
        """
        Whether to run the simplex weighted average through `embedding_bag`
        instead of gather + `bmm`.

        The fused path folds the neighbor gather into the weighted sum, so it
        never materializes the (rows, k, y_width) block. Below the threshold
        `bmm` is still the faster of the two, unless `rows` is given and that
        block would be far larger than cache.

        CUDA has no width cliff, but skipping the k-fold intermediate pays
        there too, and the target axis is left unsplit on CUDA so the width
        comes from the target count rather than from E_y. MPS keeps `bmm`;
        it has not been measured.
        """
        if not self.device.startswith(("cpu", "cuda")):
            return False
        if int(nbrs_num_max) * int(y_width) > self._BMM_FAST_ELEMS:
            return True
        if rows is None:
            return False
        staging_bytes = (
            int(rows) * int(nbrs_num_max) * int(y_width)
            * torch.tensor([], dtype=self.compute_dtype).element_size()
        )
        return staging_bytes > self._BMM_STAGING_MAX_BYTES
