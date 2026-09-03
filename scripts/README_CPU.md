# CPU Benchmark Reports

Latest benchmark snapshots written directly by the scripts in this folder.

<!-- benchmark-report:matrix-performance start -->
## Matrix Performance

Source: `benchmark_performance.py`

Settings:
- `device`: `cpu`
- `dtype`: `float32`
- `method`: `simplex`
- `tp`: `0`
- `exclusion_window`: `5`
- `library_size`: `all points`
- `sample_size`: `all points`
- `batch_size`: `auto`
- `memory_budget_gb`: `10.0`
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
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.042433 | 0.041967 | 0.042736 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.106303 | 0.103641 | 0.109104 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.362768 | 0.347928 | 0.373616 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 1.064328 | 1.040074 | 1.092325 |
<!-- benchmark-report:matrix-performance end -->

<!-- benchmark-report:single-series start -->
## Single-Series Self-Prediction

Source: `benchmark_single_series_self_prediction.py`

Settings:
- `scenario`: `single-series self-prediction`
- `device`: `cpu`
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
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.008117 | 0.007761 | 0.008326 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.078616 | 0.078194 | 0.079373 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 1.255505 | 1.246996 | 1.269540 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 20.402604 | 20.277763 | 20.603484 |
<!-- benchmark-report:single-series end -->

<!-- benchmark-report:cpu-usage start -->
## CPU Usage

Source: `benchmark_cpu_usage.py`

Settings:
- `scenario`: `cpu_usage`
- `device`: `cpu`
- `dtype`: `float32`
- `method`: `simplex`
- `tp`: `0`
- `exclusion_window`: `5`
- `library_size`: `all points`
- `sample_size`: `all points`
- `batch_size`: `auto`
- `memory_budget_gb`: `2.0`
- `xtwx_precompute`: `True`
- `xtwy_precompute`: `False`
- `attempts`: `1`
- `benchmark_cases`: `[(50, 50, 500), (50, 50, 1000), (50, 50, 4000), (50, 50, 8000), (100, 100, 500), (100, 100, 1000), (100, 100, 4000), (100, 100, 8000), (200, 200, 500), (200, 200, 1000), (200, 200, 4000), (200, 200, 8000), (400, 400, 500), (400, 400, 1000), (400, 400, 4000), (400, 400, 8000), (800, 800, 500), (800, 800, 1000), (800, 800, 4000), (800, 800, 8000)]`
- `x_embedding_dim`: `5`
- `torch_num_threads`: `10`
- `torch_num_interop_threads`: `1`
- `cpu_pct_note`: `process CPU usage, can exceed 100% when multiple CPU cores are busy`

Results:

| n_x | n_y | ts_length | ex | ey | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec | avg_cpu_sec | min_cpu_sec | max_cpu_sec | avg_cpu_pct | min_cpu_pct | max_cpu_pct | avg_cpu_cores |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 50 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.019761 | 0.019761 | 0.019761 | 0.091320 | 0.091320 | 0.091320 | 462.13 | 462.13 | 462.13 | 4.62 |
| 50 | 50 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.041021 | 0.041021 | 0.041021 | 0.218801 | 0.218801 | 0.218801 | 533.38 | 533.38 | 533.38 | 5.33 |
| 50 | 50 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.303068 | 0.303068 | 0.303068 | 1.693432 | 1.693432 | 1.693432 | 558.76 | 558.76 | 558.76 | 5.59 |
| 50 | 50 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 0.894402 | 0.894402 | 0.894402 | 5.537814 | 5.537814 | 5.537814 | 619.16 | 619.16 | 619.16 | 6.19 |
| 100 | 100 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.021629 | 0.021629 | 0.021629 | 0.135324 | 0.135324 | 0.135324 | 625.66 | 625.66 | 625.66 | 6.26 |
| 100 | 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.050829 | 0.050829 | 0.050829 | 0.345696 | 0.345696 | 0.345696 | 680.12 | 680.12 | 680.12 | 6.80 |
| 100 | 100 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.467387 | 0.467387 | 0.467387 | 3.149985 | 3.149985 | 3.149985 | 673.96 | 673.96 | 673.96 | 6.74 |
| 100 | 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 1.769348 | 1.769348 | 1.769348 | 11.271319 | 11.271319 | 11.271319 | 637.03 | 637.03 | 637.03 | 6.37 |
| 200 | 200 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.055924 | 0.055924 | 0.055924 | 0.364082 | 0.364082 | 0.364082 | 651.03 | 651.03 | 651.03 | 6.51 |
| 200 | 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.139840 | 0.139840 | 0.139840 | 0.994745 | 0.994745 | 0.994745 | 711.35 | 711.35 | 711.35 | 7.11 |
| 200 | 200 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 1.047138 | 1.047138 | 1.047138 | 6.906048 | 6.906048 | 6.906048 | 659.52 | 659.52 | 659.52 | 6.60 |
| 200 | 200 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 3.735662 | 3.735662 | 3.735662 | 23.542584 | 23.542584 | 23.542584 | 630.21 | 630.21 | 630.21 | 6.30 |
| 400 | 400 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.257059 | 0.257059 | 0.257059 | 1.560561 | 1.560561 | 1.560561 | 607.08 | 607.08 | 607.08 | 6.07 |
| 400 | 400 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.391303 | 0.391303 | 0.391303 | 2.903793 | 2.903793 | 2.903793 | 742.08 | 742.08 | 742.08 | 7.42 |
| 400 | 400 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 2.558988 | 2.558988 | 2.558988 | 16.505301 | 16.505301 | 16.505301 | 644.99 | 644.99 | 644.99 | 6.45 |
| 400 | 400 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 8.510416 | 8.510416 | 8.510416 | 51.977628 | 51.977628 | 51.977628 | 610.75 | 610.75 | 610.75 | 6.11 |
| 800 | 800 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.391898 | 0.391898 | 0.391898 | 3.023075 | 3.023075 | 3.023075 | 771.39 | 771.39 | 771.39 | 7.71 |
| 800 | 800 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 1.081611 | 1.081611 | 1.081611 | 7.668551 | 7.668551 | 7.668551 | 708.99 | 708.99 | 708.99 | 7.09 |
| 800 | 800 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 8.764998 | 8.764998 | 8.764998 | 48.476632 | 48.476632 | 48.476632 | 553.07 | 553.07 | 553.07 | 5.53 |
| 800 | 800 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 25.980612 | 25.980612 | 25.980612 | 134.707851 | 134.707851 | 134.707851 | 518.49 | 518.49 | 518.49 | 5.18 |
<!-- benchmark-report:cpu-usage end -->

<!-- benchmark-report:flat-arrays start -->
## Flat Arrays

Source: `benchmark_flat_arrays.py`

Settings:
- `scenario`: `flat_arrays`
- `device`: `cpu`
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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.244822 | 0.240515 | 0.249205 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 1.290271 | 1.279345 | 1.304882 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 5.385606 | 5.369772 | 5.404663 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 17.392626 | 17.261147 | 17.563907 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.016070 | 0.014988 | 0.017906 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.069529 | 0.066659 | 0.071910 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.281602 | 0.277433 | 0.284483 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.384756 | 0.377086 | 0.394214 |
<!-- benchmark-report:flat-arrays end -->
