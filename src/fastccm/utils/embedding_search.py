"""Sampled, corresponding-target simplex searches used by ccm_utils."""
import numpy as np
import torch

from .metrics import (
    stream_metric_state_init,
    stream_metric_state_update,
    stream_metric_state_finalize,
)


def _global_coefficients(ccm, X_lib, target_lib, candidates, groups):
    """Fit corresponding global models in batches of equal embedding width.

    Use the same intercept, normal equations, and initial regularization as
    PairwiseCCM's global baseline. Ill-conditioned systems and nearly constant
    predictions retain its scalar arithmetic, which affects correlation rounding.
    """
    beta = X_lib.new_zeros((X_lib.shape[0], X_lib.shape[2] + 1, target_lib.shape[2]))
    scalar_rows = []
    for width, rows in groups:
        features = X_lib[..., :width].index_select(0, rows)
        design = torch.cat([torch.ones_like(features[..., :1]), features], dim=-1)
        targets = target_lib.index_select(0, rows // candidates)
        gram = design.mT @ design
        rhs = design.mT @ targets
        eye = torch.eye(width + 1, device=gram.device, dtype=gram.dtype)
        fitted, info = torch.linalg.solve_ex(gram + 1e-8 * eye, rhs)
        chol, chol_info = torch.linalg.cholesky_ex(gram)
        diagonal = chol.diagonal(dim1=-2, dim2=-1).abs()
        unstable = ((info != 0) | (chol_info != 0)
                    | (diagonal.amin(-1) < 1e-3 * diagonal.amax(-1)))
        if ccm.device.startswith("cuda"):
            # CUDA correlations accumulate in float32. Avoid changing the
            # arithmetic when a large prediction mean masks its variance.
            mean = (gram[:, :1] @ fitted).squeeze(1) / design.shape[1]
            variance = (fitted * (gram @ fitted)).sum(1) / design.shape[1] - mean.square()
            unstable |= (variance < 0.25 * mean.square()).any(-1)
        if unstable.any():
            for row in torch.nonzero(unstable, as_tuple=False).flatten().tolist():
                scalar_beta = ccm._PairwiseCCM__solve_global_linear_beta(
                    design[row], targets[row])
                fitted[row] = scalar_beta
                scalar_rows.append((int(rows[row]), width, scalar_beta))
        beta[rows, :width + 1] = fitted
    return beta, scalar_rows


@torch.inference_mode()
def paired_embedding_scores(
    ccm, x, y, *, E_range, tau_range, tp_range, library_size, sample_size,
    exclusion_window, trials, seed, series_batch_size, batch_size,
    nbrs_num=None, subtract_global=False,
):
    """Return [series, horizon, tau, E], or None if the resident grid is too big.

    Only the float32 simplex/correlation path calls this helper. Keep the
    original grid-wide padding and neighbor count to preserve tie behavior.
    Sampling uses the same common suffix and device RNG as score_matrix.
    """
    device = ccm.device
    n_time, n_series = x.shape
    grid = np.array([(e, tau) for tau in tau_range for e in E_range])
    E, tau = grid.T
    g, width, horizons_count = len(grid), int(E.max()), len(tp_range)
    max_lag = int(((E - 1) * tau).max())
    common_len = n_time - int(max(tp_range)) - max_lag
    L, S = library_size, sample_size
    counts = E + 1 if nbrs_num is None else np.asarray(nbrs_num)
    if counts.ndim == 0:
        counts = np.full(g, counts.item())
    if (counts.shape != (g,) or counts.dtype.kind not in "iu" or np.any(counts < 1)):
        raise ValueError("nbrs_num must be a positive integer or one positive integer per (tau, E) candidate.")
    k_max = int(counts.max())
    if L < k_max:
        raise ValueError("library_size must provide at least the requested number of neighbors.")

    # Conservative working-set estimate: raw inputs, sampled library and its
    # augmented operand, query vectors, distance/topk scratch, gathers, and
    # metric intermediates. Like the core's budget this is an estimate, not a
    # device allocation cap. Fall back if even one resident grid cannot fit.
    budget = int(ccm.memory_budget_gb * 1024**3 * 0.7)
    raw_bytes = 4 * (len(x) + len(y))
    fixed_bytes = raw_bytes + g * (4 * L * (2 * width + 2) + 64 * horizons_count)
    query_bytes = g * (8 * L + 8 * width + k_max * (32 + 4 * horizons_count)
                       + 64 * horizons_count)
    if subtract_global:
        fixed_bytes += g * (4 * L * (2 * width + 1 + horizons_count)
                           + 4 * (width + 1) * (width + 1 + 3 * horizons_count)
                           + 64 * horizons_count)
        fixed_bytes += 4 * S * horizons_count
        query_bytes += g * (4 * (width + 1) + 64 * horizons_count)
    if fixed_bytes + query_bytes > budget:
        return None
    requested_queries = S if batch_size in (None, "auto") else min(S, batch_size)
    if series_batch_size == "auto":
        series_chunk = min(n_series, max(1, budget // (
            fixed_bytes + requested_queries * query_bytes)))
    else:
        series_chunk = n_series if series_batch_size is None else min(n_series, series_batch_size)
    if batch_size == "auto":
        query_chunk = min(S, max(1, (budget // series_chunk - fixed_bytes) // query_bytes))
    else:
        query_chunk = requested_queries
    ccm.logger.info("Embedding search batches: series=%d queries=%d candidates=%d",
                    series_chunk, query_chunk, g)
    # A previous larger call may have left scratch buffers exceeding this plan.
    retained = sum(t.numel() * t.element_size() for t in ccm._nbr_workspace.values())
    if retained > budget // 4:
        ccm._release_nbr_workspace()

    columns = np.arange(width)
    valid_np = columns[None, :] < E[:, None]
    offsets = torch.as_tensor(np.where(
        valid_np, (columns[None, :] - E[:, None] + 1) * tau[:, None], 0), device=device)
    valid = torch.as_tensor(valid_np, device=device)
    horizons = torch.as_tensor(tp_range, device=device)
    neighbors_per_candidate = torch.as_tensor(counts, device=device)
    # Draw once per trial so changing batch sizes cannot change the experiment.
    trial_indices = []
    for trial in range(trials):
        lib_gen = sample_gen = None
        if seed is not None:
            lib_gen = torch.Generator(device=device).manual_seed(int(seed) + trial)
            sample_gen = torch.Generator(device=device).manual_seed(int(seed) + trial + 1)
        lib = torch.randperm(common_len, device=device, generator=lib_gen)[:L].clone()
        sample = torch.randperm(common_len, device=device, generator=sample_gen)[:S].clone()
        trial_indices.append((lib, sample))
    output = np.empty((n_series, horizons_count, g), dtype=np.float32)
    for start in range(0, n_series, series_chunk):
        stop = min(start + series_chunk, n_series)
        raw_x = torch.as_tensor(np.ascontiguousarray(x[:, start:stop].T),
                                device=device, dtype=ccm.dtype)
        raw_y = raw_x if y is x else torch.as_tensor(
            np.ascontiguousarray(y[:, start:stop].T), device=device, dtype=ccm.dtype)
        n = stop - start
        k = neighbors_per_candidate.repeat(n)
        series_offsets = torch.arange(n, device=device)[:, None, None, None] * L
        total = torch.zeros((n, horizons_count, g), device=device, dtype=ccm.compute_dtype)
        if subtract_global:
            dimensions = np.tile(E, n)
            groups = [(int(e), torch.as_tensor(np.flatnonzero(dimensions == e), device=device))
                      for e in np.unique(E)]

        def sampled(times):
            a = raw_x[:, times[None, :, None] + offsets[:, None, :]]
            # Invalid padding must be zero even for non-finite input values.
            a.masked_fill_(~valid[None, :, None, :], 0)
            return a.reshape(n * g, len(times), width).contiguous()

        for lib_indices, sample_indices in trial_indices:
            lib_t = lib_indices + max_lag
            X_lib = sampled(lib_t)
            lib_index = ccm._prepare_nbr_library(X_lib)
            target_lib = raw_y[:, lib_t[:, None] + horizons[None, :]].contiguous()
            state = stream_metric_state_init("corr", 1, horizons_count, n * g,
                device=device, dtype=ccm.compute_dtype, shared_target=False)
            if subtract_global:
                beta, scalar_rows = _global_coefficients(ccm, X_lib, target_lib, g, groups)
                # Preserve the original reduction layout per series. This is
                # significant for nearly constant global predictions on CUDA.
                global_states = [stream_metric_state_init("corr", 1, horizons_count, g,
                    device=device, dtype=ccm.compute_dtype, shared_target=False) for _ in range(n)]
                global_targets = raw_y[:, (sample_indices + max_lag)[:, None]
                                       + horizons[None, :]].transpose(1, 2).contiguous()
            for q0 in range(0, S, query_chunk):
                sample_idx = sample_indices[q0:q0 + query_chunk]
                sample_t = sample_idx + max_lag
                q = len(sample_idx)
                X_sample = sampled(sample_t)
                weights, indices = ccm._PairwiseCCM__get_nbrs_indices_with_weights(
                    X_lib, X_sample, k, k_max, lib_indices, sample_idx,
                    exclusion_window, lib_index=lib_index)
                adjusted = indices.reshape(n, g, q, k_max) + series_offsets
                targets = torch.index_select(target_lib.reshape(n * L, horizons_count),
                                             0, adjusted.reshape(-1))
                pred = torch.bmm(weights.reshape(n * g * q, 1, k_max),
                                 targets.reshape(n * g * q, k_max, horizons_count))
                pred = pred.reshape(n * g, q, horizons_count).permute(1, 2, 0).unsqueeze(1).contiguous()
                truth = raw_y[:, sample_t[:, None] + horizons[None, :]]
                observed = truth[:, None].expand(n, g, q, horizons_count).reshape(
                    n * g, q, horizons_count).permute(1, 2, 0).unsqueeze(1).contiguous()
                stream_metric_state_update("corr", state, pred, observed)
                if subtract_global:
                    design = torch.cat([torch.ones_like(X_sample[..., :1]), X_sample], dim=-1)
                    global_pred = torch.bmm(design, beta)
                    for row, width_i, scalar_beta in scalar_rows:
                        global_pred[row] = design[row, :, :width_i + 1].contiguous() @ scalar_beta
                    global_pred = global_pred.reshape(n, g, q, horizons_count)
                    for i, global_state in enumerate(global_states):
                        A = global_pred[i].permute(1, 2, 0).unsqueeze(1).contiguous()
                        B = global_targets[i, :, q0:q0 + q].unsqueeze(-1).permute(
                            1, 2, 0).unsqueeze(-1).expand(q, 1, horizons_count, g)
                        stream_metric_state_update("corr", global_state, A, B)
                    del A, B
                    del design, global_pred
                del X_sample, weights, indices, adjusted, targets, pred, truth, observed
            score = stream_metric_state_finalize("corr", state)
            if subtract_global:
                score -= torch.cat([stream_metric_state_finalize("corr", s)
                                    for s in global_states], dim=-1)
                del beta, scalar_rows, global_states, global_targets
            total += score[0].reshape(
                horizons_count, n, g).permute(1, 0, 2)
            del X_lib, lib_index, target_lib, state
        output[start:stop] = (total / trials).cpu().numpy()
    return output.reshape(n_series, horizons_count, len(tau_range), len(E_range))
