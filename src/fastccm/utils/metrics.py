# fastccm/utils/metrics.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import torch

# All metrics take A,B with shape [S, D, Y, X] and return [D, Y, X]
Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# CPU matrix sweeps show the centered block merger paying off at this scale.
_CORR_CENTERED_MIN_SOURCES = 512


def _corr_accum_dtype(dtype: torch.dtype, device) -> torch.dtype:
    # Keep the CCM kernels in float32 when requested, but accumulate streamed
    # correlation statistics in float64 on CPU to avoid catastrophic cancellation.
    if str(device).startswith("cpu") and dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return dtype


def _double_center(D: torch.Tensor) -> torch.Tensor:
    # D: [S, S, N] where N = D*Y*X (vectorized across channels)
    mr = D.mean(dim=1, keepdim=True)            # row means [S,1,N]
    mc = D.mean(dim=0, keepdim=True)            # col means [1,S,N]
    ma = D.mean(dim=(0, 1), keepdim=True)       # grand mean [1,1,N]
    return D - mr - mc + ma


def _nan_on_constant(r: torch.Tensor, ssA: torch.Tensor, ssB: torch.Tensor) -> torch.Tensor:
    """Report NaN where a channel is constant, instead of a clean 0.

    Correlation against a constant is undefined. Targets narrower than the
    widest one in the call are zero-padded, so their padded dimensions are
    constant zero in both the prediction and the target: 0/0. The `eps` in the
    denominator resolves that to exactly 0.0, which `np.nanmean` over the
    dimension axis then averages in as though it were a real score of zero,
    silently deflating every target whose E is below the maximum. NaN is what
    callers already expect for an absent dimension -- `nanmean` skips it and
    `Visualizer.plot_convergence_test` counts dimensions with `np.isnan`.

    `ssA`/`ssB` are sums of squared deviations, so they are >= 0 and only a
    genuinely constant channel reaches 0.
    """
    return r.masked_fill((ssA <= 0) | (ssB <= 0), float("nan"))


def batch_corr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Pearson r across sample axis, keep [D,Y,X]
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    muA = A.mean(dim=0, keepdim=True)
    muB = B.mean(dim=0, keepdim=True)
    num = ((A - muA) * (B - muB)).sum(dim=0)
    ssA = ((A - muA).pow(2)).sum(dim=0)
    ssB = ((B - muB).pow(2)).sum(dim=0)
    den = torch.sqrt(ssA * ssB + eps_t)
    return _nan_on_constant((num / den).clamp(-1.0, 1.0), ssA, ssB)


def batch_mse(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A - B).pow(2).mean(dim=0)

 
def batch_mae(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE) across samples.
    A, B: [S, D, Y, X]  ->  returns [D, Y, X]
    """
    return (A - B).abs().mean(dim=0)


def batch_rmse(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    return torch.sqrt(batch_mse(A, B) + eps_t)


def batch_neg_nrmse(A: torch.Tensor, B: torch.Tensor,
                                 T: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """
    Exponentiated normalized squared error across samples (S) and features (D).
    The baseline predicts each feature's sample mean separately at each (y,x).
    This preserves variance weighting and invariance to a shared translation
    and full orthogonal feature transform (including PCA without whitening).
    A,B: [S, D, Y, X]
    Returns: [1, Y, X]
    """
    T_t  = torch.tensor(T,  dtype=A.dtype, device=A.device)
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    # RMSE over (S, D) per (y, x)
    mse  = (A - B).pow(2).mean(dim=(0, 1))              # [Y, X]
    rmse = torch.sqrt(mse + eps_t)                         # [Y, X]

    # Baseline: each feature's mean over samples.
    muB  = B.mean(dim=0, keepdim=True)                 # [1, D, Y, X]
    varB = (B - muB).pow(2).mean(dim=(0, 1))            # [Y, X]
    rmse_base = torch.sqrt(varB + eps_t)                  # [Y, X]

    neg_nrmse = torch.exp(- ((1.0 / T_t) * torch.pow(rmse / (rmse_base + eps_t), 2)))  # [Y, X]
    return neg_nrmse.unsqueeze(0).to(dtype=A.dtype)        # [1, Y, X]

def batch_dcor(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Multivariate distance correlation across samples using ALL features D,
    computed separately for each (y, x).

    Inputs:
        A, B: [S, D, Y, X]  (their dtype controls compute precision)
    Returns:
        dCor broadcast to [D, Y, X] (same value along D for each (y, x))
    """
    wd = A.dtype
    eps_t = torch.tensor(eps, dtype=wd, device=A.device)

    S, D, Y, X = A.shape

    # reshape to [Y*X, S, D]
    A2 = A.permute(2, 3, 0, 1).reshape(-1, S, D).to(wd)
    B2 = B.permute(2, 3, 0, 1).reshape(-1, S, D).to(wd)

    # pairwise distances
    DA = torch.cdist(A2, A2, p=2, compute_mode="use_mm_for_euclid_dist")
    DB = torch.cdist(B2, B2, p=2, compute_mode="use_mm_for_euclid_dist")

    # double-centering per block
    def _dc(M):
        mr = M.mean(dim=-1, keepdim=True)
        mc = M.mean(dim=-2, keepdim=True)
        ma = M.mean(dim=(-1, -2), keepdim=True)
        return M - mr - mc + ma

    A_dc, B_dc = _dc(DA), _dc(DB)

    # dCov / sqrt(dVarA * dVarB) per (y,x)
    dCov   = (A_dc * B_dc).mean(dim=(-1, -2))
    dVar_A = (A_dc.pow(2)).mean(dim=(-1, -2))
    dVar_B = (B_dc.pow(2)).mean(dim=(-1, -2))
    dCor   = dCov / (torch.sqrt(dVar_A * dVar_B) + eps_t)        

    # reshape back and broadcast
    dCor_yx = dCor.reshape(Y, X).to(dtype=A.dtype)
    return dCor_yx.unsqueeze(0).expand(D, Y, X)

# ---- registry ----
_METRICS: Dict[str, Metric] = {
    "corr":      batch_corr,
    "mse":       batch_mse,
    "mae":       batch_mae,
    "rmse":      batch_rmse,
    "neg_nrmse": batch_neg_nrmse,
    "dcorr":     batch_dcor,
}

def get_metric(name: str) -> Metric:
    if name not in _METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(_METRICS)}")
    return _METRICS[name]


def get_streaming_metric_kind(metric_fn) -> Optional[str]:
    name = getattr(metric_fn, "__name__", "")
    return {
        "batch_corr": "corr",
        "batch_mse": "mse",
        "batch_rmse": "rmse",
        "batch_mae": "mae",
        "batch_neg_nrmse": "neg_nrmse",
    }.get(name)


def stream_metric_state_init(kind: str, D, Y, X, *, device, dtype, shared_target: bool = False):
    shape_dyx = (D, Y, X)
    out_dtype = dtype
    acc_dtype = _corr_accum_dtype(dtype, device) if kind == "corr" else dtype
    z_dyx = torch.zeros(shape_dyx, device=device, dtype=acc_dtype)
    state = {
        "n": 0,
        "D": int(D),
        "dtype": out_dtype,
        "acc_dtype": acc_dtype,
        "device": device,
        "shared_target": bool(shared_target),
    }

    if kind == "corr":
        target_x = 1 if shared_target else X
        z_dyb = torch.zeros((D, Y, target_x), device=device, dtype=acc_dtype)
        centered = acc_dtype != dtype and int(X) >= _CORR_CENTERED_MIN_SOURCES
        state["corr_centered"] = centered
        if centered:
            state.update({
                "meanA": z_dyx.clone(),
                "meanB": z_dyb.clone(),
                "m2A": z_dyx.clone(),
                "m2B": z_dyb.clone(),
                "coAB": z_dyx.clone(),
            })
        else:
            state.update({
                "sumA": z_dyx.clone(),
                "sumB": z_dyb.clone(),
                "sumAA": z_dyx.clone(),
                "sumBB": z_dyb.clone(),
                "sumAB": z_dyx.clone(),
            })
        return state
    if kind in ("mse", "rmse"):
        state["sum_sq_err"] = z_dyx
        return state
    if kind == "mae":
        state["sum_abs_err"] = z_dyx
        return state
    if kind == "neg_nrmse":
        shape_yx = (Y, X)
        z_yx = torch.zeros(shape_yx, device=device, dtype=dtype)
        target_x = 1 if shared_target else X
        z_dyb = torch.zeros((D, Y, target_x), device=device, dtype=dtype)
        state.update({
            "sum_sq_err_sd": z_yx.clone(),  # over S and D
            "meanB": z_dyb.clone(),         # per-feature sample mean
            "m2B": z_dyb.clone(),           # centered sum of squares over S
        })
        return state
    raise ValueError(f"Unsupported streaming metric kind: {kind}")


def stream_metric_state_update(kind: str, state, A_blk, B_blk, *, y_start: int = 0, count_samples: bool = True):
    if count_samples:
        state["n"] += int(A_blk.shape[0])
    y_stop = int(y_start + A_blk.shape[2])
    dyx = (slice(None), slice(int(y_start), y_stop), slice(None))
    dy1 = (slice(None), slice(int(y_start), y_stop), slice(None))
    yx = (slice(int(y_start), y_stop), slice(None))
    y1 = (slice(int(y_start), y_stop), slice(None))

    if kind == "corr":
        if state.get("shared_target", False):
            B_blk = B_blk[..., :1]
        if state.get("corr_centered", False):
            # Center in float32, then merge only reduced summaries in float64.
            # This avoids a full promoted prediction tile without sacrificing
            # stability across sample batches with different means.
            work_dtype = (
                torch.float32
                if A_blk.dtype in (torch.float16, torch.bfloat16)
                else A_blk.dtype
            )
            A_work = A_blk.to(dtype=work_dtype)
            B_work = B_blk.to(dtype=work_dtype)
            anchorA = A_work[:1]
            anchorB = B_work[:1]
            deltaA_work = A_work - anchorA
            deltaB_work = B_work - anchorB
            mean_deltaA = deltaA_work.mean(dim=0)
            mean_deltaB = deltaB_work.mean(dim=0)
            centeredA = deltaA_work - mean_deltaA
            centeredB = deltaB_work - mean_deltaB
            m2A_blk = (centeredA * centeredA).sum(dim=0).to(state["acc_dtype"])
            m2B_blk = (centeredB * centeredB).sum(dim=0).to(state["acc_dtype"])
            coAB_blk = (centeredA * centeredB).sum(dim=0).to(state["acc_dtype"])
            meanA_blk = anchorA[0].to(state["acc_dtype"]) + mean_deltaA.to(state["acc_dtype"])
            meanB_blk = anchorB[0].to(state["acc_dtype"]) + mean_deltaB.to(state["acc_dtype"])

            n_blk = int(A_blk.shape[0])
            n_total = int(state["n"])
            n_prev = n_total - n_blk
            if n_prev <= 0:
                state["meanA"][dyx].copy_(meanA_blk)
                state["meanB"][dy1].copy_(meanB_blk)
                state["m2A"][dyx].copy_(m2A_blk)
                state["m2B"][dy1].copy_(m2B_blk)
                state["coAB"][dyx].copy_(coAB_blk)
            else:
                merge_weight = float(n_blk) / float(n_total)
                merge_factor = float(n_prev * n_blk) / float(n_total)
                deltaA = meanA_blk - state["meanA"][dyx]
                deltaB = meanB_blk - state["meanB"][dy1]
                state["coAB"][dyx] += coAB_blk + deltaA * deltaB * merge_factor
                state["m2A"][dyx] += m2A_blk + deltaA * deltaA * merge_factor
                state["m2B"][dy1] += m2B_blk + deltaB * deltaB * merge_factor
                state["meanA"][dyx] += deltaA * merge_weight
                state["meanB"][dy1] += deltaB * merge_weight
        else:
            A_blk = A_blk.to(device=state["device"], dtype=state["acc_dtype"])
            B_blk = B_blk.to(device=state["device"], dtype=state["acc_dtype"])
            state["sumA"][dyx] += A_blk.sum(dim=0)
            state["sumB"][dy1] += B_blk.sum(dim=0)
            state["sumAA"][dyx] += (A_blk * A_blk).sum(dim=0)
            state["sumBB"][dy1] += (B_blk * B_blk).sum(dim=0)
            state["sumAB"][dyx] += (A_blk * B_blk).sum(dim=0)
        return
    if kind in ("mse", "rmse"):
        d = A_blk - B_blk
        state["sum_sq_err"][dyx] += (d * d).sum(dim=0)
        return
    if kind == "mae":
        state["sum_abs_err"][dyx] += (A_blk - B_blk).abs().sum(dim=0)
        return
    if kind == "neg_nrmse":
        block_n = int(A_blk.shape[0])
        if block_n == 0:
            return
        d = A_blk - B_blk
        state["sum_sq_err_sd"][yx] += (d * d).sum(dim=(0, 1))
        if state.get("shared_target", False):
            B_blk = B_blk[..., :1]
        block_mean = B_blk.mean(dim=0)
        centered = B_blk - block_mean.unsqueeze(0)
        delta = block_mean - state["meanB"][dy1]
        old_n = state["n"] - block_n
        # Merge centered moments over all sample chunks, retaining variation
        # between chunk means. Target tiles share the same total sample count.
        state["m2B"][dy1] += (centered.square().sum(dim=0)
                              + delta.square() * (old_n * block_n / state["n"]))
        state["meanB"][dy1] += delta * (block_n / state["n"])
        return
    raise ValueError(f"Unsupported streaming metric kind: {kind}")


def stream_metric_state_finalize(kind: str, state, *, eps=1e-12, neg_nrmse_T=0.5):
    n = max(int(state["n"]), 1)
    D = max(int(state["D"]), 1)
    device = state["device"]
    out_dtype = state["dtype"]
    acc_dtype = state.get("acc_dtype", out_dtype)
    n_t = torch.tensor(float(n), device=device, dtype=acc_dtype)
    eps_t = torch.tensor(eps, device=device, dtype=acc_dtype)

    if kind == "corr":
        if state.get("corr_centered", False):
            num = state["coAB"]
            denA = state["m2A"]
            denB = state["m2B"]
        else:
            num = state["sumAB"] - (state["sumA"] * state["sumB"] / n_t)
            denA = state["sumAA"] - (state["sumA"] * state["sumA"] / n_t)
            denB = state["sumBB"] - (state["sumB"] * state["sumB"] / n_t)
        denA = denA.clamp_min(0.0)
        denB = denB.clamp_min(0.0)
        den = torch.sqrt(denA * denB + eps_t)
        out = _nan_on_constant((num / den).clamp(-1.0, 1.0), denA, denB)
        return out.to(dtype=out_dtype)
    if kind == "mse":
        return state["sum_sq_err"] / n_t
    if kind == "rmse":
        return torch.sqrt((state["sum_sq_err"] / n_t) + eps_t)
    if kind == "mae":
        return state["sum_abs_err"] / n_t
    if kind == "neg_nrmse":
        cnt_t = torch.tensor(float(n * D), device=device, dtype=out_dtype)
        mse = state["sum_sq_err_sd"] / cnt_t
        rmse = torch.sqrt(mse + eps_t)
        varB = state["m2B"].sum(dim=0) / cnt_t
        rmse_base = torch.sqrt(varB.clamp_min(0.0) + eps_t)
        T_t = torch.tensor(neg_nrmse_T, device=device, dtype=out_dtype)
        out = torch.exp(-((1.0 / T_t) * torch.pow(rmse / (rmse_base + eps_t), 2)))
        return out.unsqueeze(0).to(dtype=out_dtype)
    raise ValueError(f"Unsupported streaming metric kind: {kind}")
