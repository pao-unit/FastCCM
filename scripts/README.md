# Benchmark Reports

Latest benchmark snapshots written directly by the scripts in this folder.

<!-- benchmark-report:matrix-performance start -->
## Matrix Performance

Source: `benchmark_performance.py`

Settings:
- `device`: `cuda`
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
- `torch_num_threads`: `32`
- `torch_num_interop_threads`: `1`

Results:

| matrix_size | ts_length | ex | ey | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.023427 | 0.022992 | 0.023930 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.045167 | 0.043536 | 0.046919 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.149560 | 0.147257 | 0.152447 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 0.255305 | 0.254659 | 0.256351 |
<!-- benchmark-report:matrix-performance end -->

<!-- benchmark-report:flat-arrays start -->
## Flat Arrays

Source: `benchmark_flat_arrays.py`

Settings:
- `scenario`: `flat_arrays`
- `device`: `cuda`
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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.076025 | 0.071962 | 0.078860 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.397972 | 0.389591 | 0.410001 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 1.583048 | 1.540807 | 1.641421 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 4.555152 | 4.530931 | 4.567946 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.013020 | 0.012322 | 0.013565 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.047323 | 0.045840 | 0.048956 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.184619 | 0.176933 | 0.191249 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.272983 | 0.258270 | 0.286246 |
<!-- benchmark-report:flat-arrays end -->

<!-- benchmark-report:single-series start -->
## Single-Series Self-Prediction

Source: `benchmark_single_series_self_prediction.py`

Settings:
- `scenario`: `single-series self-prediction`
- `device`: `cuda`
- `dtype`: `float32`
- `method`: `simplex`
- `E`: `20`
- `tau`: `1`
- `tp`: `1`
- `exclusion_window`: `10`
- `library_size`: `all valid points`
- `sample_size`: `all valid points`
- `batch_size`: `auto`
- `memory_budget_gb`: `16.0`
- `xtwx_precompute`: `True`
- `xtwy_precompute`: `True`
- `attempts`: `3`
- `lengths`: `[2000, 8000, 32000, 128000]`
- `torch_num_threads`: `32`
- `torch_num_interop_threads`: `1`

Results:

| length | embedded_length | library_size | sample_size | exclusion_window | attempts | avg_sec | min_sec | max_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.434323 | 0.004815 | 1.293321 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.008351 | 0.007528 | 0.009386 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 0.053833 | 0.049815 | 0.058614 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 0.746251 | 0.743465 | 0.751764 |
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
| 50 | 50 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.037903 | 0.037903 | 0.037903 | 0.138070 | 0.138070 | 0.138070 | 364.27 | 364.27 | 364.27 | 3.64 |
| 50 | 50 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.051845 | 0.051845 | 0.051845 | 0.257462 | 0.257462 | 0.257462 | 496.60 | 496.60 | 496.60 | 4.97 |
| 50 | 50 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.443639 | 0.443639 | 0.443639 | 2.056963 | 2.056963 | 2.056963 | 463.66 | 463.66 | 463.66 | 4.64 |
| 50 | 50 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 1.689854 | 1.689854 | 1.689854 | 7.292554 | 7.292554 | 7.292554 | 431.55 | 431.55 | 431.55 | 4.32 |
| 100 | 100 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.035387 | 0.035387 | 0.035387 | 0.182016 | 0.182016 | 0.182016 | 514.35 | 514.35 | 514.35 | 5.14 |
| 100 | 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.091968 | 0.091968 | 0.091968 | 0.475492 | 0.475492 | 0.475492 | 517.02 | 517.02 | 517.02 | 5.17 |
| 100 | 100 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.873551 | 0.873551 | 0.873551 | 4.122817 | 4.122817 | 4.122817 | 471.96 | 471.96 | 471.96 | 4.72 |
| 100 | 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 3.307520 | 3.307520 | 3.307520 | 15.027218 | 15.027218 | 15.027218 | 454.33 | 454.33 | 454.33 | 4.54 |
| 200 | 200 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.086549 | 0.086549 | 0.086549 | 0.491749 | 0.491749 | 0.491749 | 568.17 | 568.17 | 568.17 | 5.68 |
| 200 | 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.217008 | 0.217008 | 0.217008 | 1.252687 | 1.252687 | 1.252687 | 577.25 | 577.25 | 577.25 | 5.77 |
| 200 | 200 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 1.907633 | 1.907633 | 1.907633 | 9.052553 | 9.052553 | 9.052553 | 474.54 | 474.54 | 474.54 | 4.75 |
| 200 | 200 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 7.003386 | 7.003386 | 7.003386 | 33.503508 | 33.503508 | 33.503508 | 478.39 | 478.39 | 478.39 | 4.78 |
| 400 | 400 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.239758 | 0.239758 | 0.239758 | 1.591149 | 1.591149 | 1.591149 | 663.65 | 663.65 | 663.65 | 6.64 |
| 400 | 400 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.606482 | 0.606482 | 0.606482 | 3.759656 | 3.759656 | 3.759656 | 619.91 | 619.91 | 619.91 | 6.20 |
| 400 | 400 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 4.604096 | 4.604096 | 4.604096 | 24.589008 | 24.589008 | 24.589008 | 534.07 | 534.07 | 534.07 | 5.34 |
| 400 | 400 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 15.893523 | 15.893523 | 15.893523 | 79.290982 | 79.290982 | 79.290982 | 498.89 | 498.89 | 498.89 | 4.99 |
| 800 | 800 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 1.218005 | 1.218005 | 1.218005 | 9.046503 | 9.046503 | 9.046503 | 742.73 | 742.73 | 742.73 | 7.43 |
| 800 | 800 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 1.975341 | 1.975341 | 1.975341 | 13.578030 | 13.578030 | 13.578030 | 687.38 | 687.38 | 687.38 | 6.87 |
| 800 | 800 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 12.246079 | 12.246079 | 12.246079 | 67.676433 | 67.676433 | 67.676433 | 552.64 | 552.64 | 552.64 | 5.53 |
| 800 | 800 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 37.514336 | 37.514336 | 37.514336 | 191.501151 | 191.501151 | 191.501151 | 510.47 | 510.47 | 510.47 | 5.10 |
<!-- benchmark-report:cpu-usage end -->
