# GPU Benchmark Reports

Latest benchmark snapshots written directly by the scripts in this folder.

<!-- benchmark-report:flat-arrays start -->
## Flat Arrays

Source: `benchmark_flat_arrays.py`

Settings:
- `scenario`: `flat_arrays`
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.3.1+cu121`
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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.052142 | 0.050645 | 0.053613 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.280610 | 0.278620 | 0.282428 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 1.123643 | 1.122794 | 1.125166 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 3.365427 | 3.352371 | 3.385158 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.003745 | 0.003718 | 0.003783 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.015669 | 0.015404 | 0.016088 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.085141 | 0.080886 | 0.093162 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.096786 | 0.083896 | 0.103363 |
<!-- benchmark-report:flat-arrays end -->

<!-- benchmark-report:matrix-performance start -->
## Matrix Performance

Source: `benchmark_performance.py`

Settings:
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.3.1+cu121`
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
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.007949 | 0.007545 | 0.008507 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.016505 | 0.016035 | 0.016976 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.049955 | 0.049721 | 0.050330 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 0.208710 | 0.208559 | 0.208969 |
<!-- benchmark-report:matrix-performance end -->

<!-- benchmark-report:single-series start -->
## Single-Series Self-Prediction

Source: `benchmark_single_series_self_prediction.py`

Settings:
- `scenario`: `single-series self-prediction`
- `device`: `cuda`
- `gpu`: `NVIDIA H100 PCIe`
- `torch`: `2.3.1+cu121`
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
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.001613 | 0.001481 | 0.001842 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.006702 | 0.006680 | 0.006717 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 0.082232 | 0.081514 | 0.082713 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 1.384353 | 1.377742 | 1.391032 |
<!-- benchmark-report:single-series end -->
