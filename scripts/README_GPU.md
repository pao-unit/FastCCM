# GPU Benchmark Reports

Latest benchmark snapshots written directly by the scripts in this folder.

<!-- benchmark-report:flat-arrays start -->
## Flat Arrays

Source: `benchmark_flat_arrays.py`

Settings:
- `scenario`: `flat_arrays`
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.5.1+cu121`
- `dtype`: `float32`
- `method`: `simplex`
- `tp`: `0`
- `exclusion_window`: `5`
- `library_size`: `all points`
- `sample_size`: `all points`
- `batch_size`: `auto`
- `memory_budget_gb`: `16.0`
- `xtwx_precompute`: `True`
- `xtwy_precompute`: `False`
- `attempts`: `3`
- `warmups`: `1`
- `benchmark_cases`: `[(1000, 1, 1000), (5000, 1, 1000), (20000, 1, 1000), (20000, 1, 2000), (1, 1000, 1000), (1, 5000, 1000), (1, 20000, 1000), (1, 20000, 2000)]`
- `x_embedding_dim`: `5`
- `torch_num_threads`: `10`
- `torch_num_interop_threads`: `1`

Results:

| n_x | n_y | ts_length | ex | ey | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.045312 | 0.044795 | 0.046033 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.245397 | 0.242002 | 0.251935 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.971647 | 0.969720 | 0.973339 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 2.682948 | 2.664168 | 2.720310 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.005361 | 0.004885 | 0.005914 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.021152 | 0.019954 | 0.022942 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.083439 | 0.077775 | 0.094189 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.112280 | 0.103756 | 0.127150 |
<!-- benchmark-report:flat-arrays end -->

<!-- benchmark-report:matrix-performance start -->
## Matrix Performance

Source: `benchmark_performance.py`

Settings:
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.5.1+cu121`
- `dtype`: `float32`
- `method`: `simplex`
- `tp`: `0`
- `exclusion_window`: `5`
- `library_size`: `all points`
- `sample_size`: `all points`
- `batch_size`: `auto`
- `memory_budget_gb`: `16.0`
- `xtwx_precompute`: `True`
- `xtwy_precompute`: `False`
- `attempts`: `3`
- `warmups`: `1`
- `matrix_time_pairs`: `[(100, 1000), (200, 1000), (800, 500), (100, 8000)]`
- `x_embedding_dim`: `5`
- `torch_num_threads`: `10`
- `torch_num_interop_threads`: `1`

Results:

| matrix_size | ts_length | ex | ey | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.006838 | 0.006599 | 0.007241 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.013074 | 0.012861 | 0.013468 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.036624 | 0.036386 | 0.037025 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 0.083923 | 0.083674 | 0.084260 |
<!-- benchmark-report:matrix-performance end -->

<!-- benchmark-report:single-series start -->
## Single-Series Self-Prediction

Source: `benchmark_single_series_self_prediction.py`

Settings:
- `scenario`: `single-series self-prediction`
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.5.1+cu121`
- `dtype`: `float32`
- `method`: `smap`
- `E`: `20`
- `tau`: `1`
- `tp`: `1`
- `exclusion_window`: `10`
- `library_size`: `all valid points`
- `sample_size`: `all valid points`
- `batch_size`: `auto`
- `memory_budget_gb`: `10.0`
- `xtwx_precompute`: `True`
- `xtwy_precompute`: `True`
- `attempts`: `3`
- `warmups`: `1`
- `lengths`: `[2000, 8000, 32000, 128000]`
- `torch_num_threads`: `10`
- `torch_num_interop_threads`: `1`

Results:

| length | embedded_length | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.001595 | 0.001465 | 0.001842 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.006792 | 0.006741 | 0.006818 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 0.082072 | 0.081815 | 0.082525 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 1.391188 | 1.385473 | 1.395771 |
<!-- benchmark-report:single-series end -->
