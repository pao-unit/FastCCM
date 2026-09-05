# _smap.py
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
    smap_xtwx_precompute_bytes,
    smap_xtwy_precompute_bytes,
    tic,
    time_block,
    timings_summary,
    toc_ms,
)

class _SmapMixin:
    """Locally weighted linear prediction (S-Map).

    Reaches outside itself only for the distance helper."""

    @torch.inference_mode()
    def _smap_prediction(self, lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted,
                      exclusion_rad, theta, metric_fn, return_pred=False,
                      sample_batch_size=None, ridge=0.0,
                      xtwx_precompute=True, xtwy_precompute=False):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_X  = X_lib.shape[2]
        max_E_Y  = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]
        subset_size    = X_lib.shape[1]
        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device=self.device, dtype=self.compute_dtype
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        A_all = None if stream_kind is not None else torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=out_device, dtype=self.dtype)

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        self.logger.debug(
            "Entering smap backend (queries=%d, library_points=%d, num_sources=%d, num_targets=%d, Ex=%d, Ey=%d, batch_size=%d, num_batches=%d, exclusion=%s, theta=%s, ridge=%s, xtwx_precompute=%s, xtwy_precompute=%s)",
            int(subsample_size),
            int(subset_size),
            int(num_ts_X),
            int(num_ts_Y),
            int(max_E_X),
            int(max_E_Y),
            int(sample_batch_size),
            int(math.ceil(subsample_size / sample_batch_size)),
            str(exclusion_rad),
            str(theta),
            str(ridge),
            str(xtwx_precompute),
            str(xtwy_precompute),
        )

        Xc = X_lib.to(self.compute_dtype)                 # (nX, L, Ex)
        Yc = Y_lib_shifted.to(self.compute_dtype)         # (nY, L, Ey)
        onesL = torch.ones((num_ts_X, subset_size, 1), device=self.device, dtype=self.compute_dtype)
        Xint = torch.cat([onesL, Xc], dim=2)             # (nX, L, Ex1)
        Xint_t = Xint.transpose(1, 2).contiguous()       # (nX, Ex1, L)
        Yc_flat = Yc.permute(1, 0, 2).reshape(subset_size, num_ts_Y * max_E_Y).contiguous()
        ex1 = int(max_E_X + 1)

        XTWX_features = None
        if xtwx_precompute:
            self.logger.debug(
                "Building XTWX precompute features (~%s)",
                format_bytes(smap_xtwx_precompute_bytes(X_lib, compute_dtype=self.compute_dtype)),
            )
            XTWX_features = torch.matmul(Xint.unsqueeze(-1), Xint.unsqueeze(-2)).reshape(
                num_ts_X, subset_size, ex1 * ex1
            )
        XTWy_features = None
        if xtwy_precompute:
            self.logger.debug(
                "Building XTWy precompute features (~%s)",
                format_bytes(smap_xtwy_precompute_bytes(X_lib, Y_lib_shifted, compute_dtype=self.compute_dtype)),
            )
            XTWy_features = torch.mul(
                Xint.unsqueeze(-1),
                Yc_flat.unsqueeze(0).unsqueeze(2),
            ).reshape(num_ts_X, subset_size, ex1 * num_ts_Y * max_E_Y)
        I = None
        if ridge and ridge > 0.0:
            I = torch.eye(ex1, device=self.device, dtype=self.compute_dtype)[None, None]
        weighted_design_buf = None
        tail_batch_size = subsample_size % sample_batch_size
        weighted_design_tail_buf = None
        if not xtwy_precompute:
            weighted_design_buf = torch.empty(
                (num_ts_X, sample_batch_size, ex1, subset_size),
                device=self.device,
                dtype=self.compute_dtype,
            )
        if (not xtwy_precompute) and tail_batch_size:
            weighted_design_tail_buf = torch.empty(
                (num_ts_X, tail_batch_size, ex1, subset_size),
                device=self.device,
                dtype=self.compute_dtype,
            )

        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "smap batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            B  = s1 - s0
            self.logger.debug(
                "SMAP batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = tic(self.logger, self.device) if self._debug_enabled() else None

            # Slice the queries 
            with time_block(self.logger, self.device, timings, "slice"):
                X_sample_b = X_sample[:, s0:s1, :].to(device=self.device, dtype=self.dtype, copy=False)  # (num_ts_X, B, max_E_X)

            try:
                with time_block(self.logger, self.device, timings, "local_weights"):
                    weights = self._get_local_weights(
                        lib=X_lib, sublib=X_sample_b,
                        subset_idx=lib_indices, sample_idx=smpl_indices[s0:s1],
                        exclusion_rad=exclusion_rad, theta=theta
                    )

                with time_block(self.logger, self.device, timings, "square"):
                    weights.square_()  # (nX, B, L) in-place; avoid extra w2 allocation

                # XTWX: (nX, B, Ex1, Ex1)
                with time_block(self.logger, self.device, timings, "XTWX"):
                    if xtwx_precompute:
                        XTWX = torch.matmul(weights, XTWX_features).reshape(num_ts_X, B, ex1, ex1)
                    else:
                        XTWX = torch.einsum("xli,xbl,xlj->xbij", Xint, weights, Xint)
                    if I is not None:
                        XTWX = XTWX + ridge * I

                # XTWy: (nX, B, Ex1, nY, Ey) -> flatten to (nX, B, Ex1, nY*Ey)
                with time_block(self.logger, self.device, timings, "XTWy"):
                    if xtwy_precompute:
                        XTWy = torch.matmul(weights, XTWy_features).reshape(num_ts_X, B, ex1, num_ts_Y * max_E_Y)
                    else:
                        weighted_design = weighted_design_buf if B == sample_batch_size else weighted_design_tail_buf
                        torch.mul(weights.unsqueeze(-2), Xint_t.unsqueeze(1), out=weighted_design)
                        XTWy = torch.matmul(weighted_design, Yc_flat)

                with time_block(self.logger, self.device, timings, "solve"):
                    beta = torch.linalg.solve(XTWX, XTWy)               # (nX, B, Ex1, nY*Ey)

                with time_block(self.logger, self.device, timings, "query_design"):
                    Xq = X_sample_b.to(self.compute_dtype)              # (nX, B, Ex)
                    Xq = torch.cat(
                        [torch.ones((num_ts_X, B, 1), device=self.device, dtype=self.compute_dtype), Xq],
                        dim=2
                    )                                                   # (nX, B, Ex1)

                with time_block(self.logger, self.device, timings, "predict"):
                    pred_flat = torch.matmul(Xq.unsqueeze(2), beta).squeeze(2)  # (nX, B, nY*Ey)

                A = pred_flat.view(num_ts_X, B, num_ts_Y, max_E_Y).permute(1, 3, 2, 0)  # (B,Ey,nY,nX)
                if stream_kind is not None:
                    with time_block(self.logger, self.device, timings, "metric"):
                        B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(device=self.device, dtype=self.compute_dtype)[:, :, :, None] \
                            .expand(B, max_E_Y, num_ts_Y, num_ts_X)
                        stream_metric_state_update(
                            stream_kind,
                            stream_state,
                            A.to(device=self.device, dtype=self.compute_dtype),
                            B_blk,
                        )
                        del B_blk
                else:
                    with time_block(self.logger, self.device, timings, "store"):
                        A_all[s0:s1] = A.to(out_device, dtype=self.dtype)

                if self._debug_enabled():
                    timings["total"] = toc_ms(self.logger, self.device, t_batch)
                    self.logger.debug(
                        "SMAP batch [%d:%d) timings: %s",
                        int(s0),
                        int(s1),
                        timings_summary(
                            timings,
                            order=["slice", "local_weights", "square", "cast_xy", "design", "XTWX", "XTWy",
                                   "solve", "query_design", "predict", "metric", "store", "total"],
                        ),
                    )

                del weights, XTWX, XTWy, beta, Xq, pred_flat, A
            except RuntimeError as e:
                if is_oom_error(e):
                    self.logger.warning("OOM in SMAP batch [%d:%d): %s", int(s0), int(s1), str(e))
                    hard_clear(self.logger, self.device)
                raise

        if stream_kind is not None:
            return stream_metric_state_finalize(stream_kind, stream_state)

        if (Y_sample_shifted is None) and return_pred:
            return A_all

        self.logger.debug("Computing SMAP score metric")
        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(
            subsample_size, max_E_Y, num_ts_Y, num_ts_X
        ).to(device=A_all.device, dtype=self.compute_dtype)
        r_AB = metric_fn(A_all.to(dtype=self.compute_dtype), B_full)

        if return_pred:
            return (r_AB, A_all)
        else:
            return r_AB

    def _get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = self._cdist(sublib, lib)
        if theta == None:
            weights = dist.neg_().exp_()
        else:
            denom = dist.mean(dim=2, keepdim=True).clamp_min(1e-12)  # (n_X, S, 1)
            weights = dist.mul_(-theta).div_(denom).exp_() 

        #if exclusion_rad > 0:
        if exclusion_rad is not None:
            #allowed = (torch.abs(subset_idx[None] - sample_idx[:,None]) > exclusion_rad)
            allowed = (
                    (subset_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
                    (subset_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
                ) 
            weights.masked_fill_(~allowed.unsqueeze(0), 0.0)

        return weights
