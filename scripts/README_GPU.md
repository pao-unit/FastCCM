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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.043188 | 0.042537 | 0.044213 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.241943 | 0.241046 | 0.243509 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.967660 | 0.966785 | 0.968698 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 2.665463 | 2.662756 | 2.670238 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.004095 | 0.004082 | 0.004105 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.016988 | 0.016833 | 0.017250 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.074396 | 0.072968 | 0.075407 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.107627 | 0.106413 | 0.108497 |
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
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.006854 | 0.006760 | 0.007018 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.014156 | 0.014012 | 0.014427 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.047333 | 0.047125 | 0.047740 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 0.145224 | 0.145116 | 0.145383 |
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
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.001639 | 0.001481 | 0.001942 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.006763 | 0.006739 | 0.006792 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 0.082108 | 0.081050 | 0.082920 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 1.380606 | 1.367311 | 1.390385 |
<!-- benchmark-report:single-series end -->
