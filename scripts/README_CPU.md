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
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.039294 | 0.038752 | 0.040012 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.100400 | 0.099060 | 0.102141 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.349728 | 0.335325 | 0.365437 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 1.026142 | 1.016932 | 1.035080 |
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
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.007651 | 0.007357 | 0.008012 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.076626 | 0.076308 | 0.076991 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 1.241246 | 1.233601 | 1.255186 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 20.414180 | 20.075255 | 20.584464 |
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
| 50 | 50 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.017129 | 0.017129 | 0.017129 | 0.074395 | 0.074395 | 0.074395 | 434.33 | 434.33 | 434.33 | 4.34 |
| 50 | 50 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.035759 | 0.035759 | 0.035759 | 0.176701 | 0.176701 | 0.176701 | 494.14 | 494.14 | 494.14 | 4.94 |
| 50 | 50 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.238341 | 0.238341 | 0.238341 | 1.173273 | 1.173273 | 1.173273 | 492.27 | 492.27 | 492.27 | 4.92 |
| 50 | 50 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 0.636445 | 0.636445 | 0.636445 | 3.529770 | 3.529770 | 3.529770 | 554.61 | 554.61 | 554.61 | 5.55 |
| 100 | 100 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.023293 | 0.023293 | 0.023293 | 0.163843 | 0.163843 | 0.163843 | 703.40 | 703.40 | 703.40 | 7.03 |
| 100 | 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.049719 | 0.049719 | 0.049719 | 0.354089 | 0.354089 | 0.354089 | 712.18 | 712.18 | 712.18 | 7.12 |
| 100 | 100 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.359341 | 0.359341 | 0.359341 | 2.184035 | 2.184035 | 2.184035 | 607.79 | 607.79 | 607.79 | 6.08 |
| 100 | 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 1.251941 | 1.251941 | 1.251941 | 7.039370 | 7.039370 | 7.039370 | 562.28 | 562.28 | 562.28 | 5.62 |
| 200 | 200 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.057405 | 0.057405 | 0.057405 | 0.366988 | 0.366988 | 0.366988 | 639.30 | 639.30 | 639.30 | 6.39 |
| 200 | 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.137901 | 0.137901 | 0.137901 | 1.005584 | 1.005584 | 1.005584 | 729.21 | 729.21 | 729.21 | 7.29 |
| 200 | 200 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.858312 | 0.858312 | 0.858312 | 4.829775 | 4.829775 | 4.829775 | 562.71 | 562.71 | 562.71 | 5.63 |
| 200 | 200 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 3.194730 | 3.194730 | 3.194730 | 15.110281 | 15.110281 | 15.110281 | 472.98 | 472.98 | 472.98 | 4.73 |
| 400 | 400 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.263617 | 0.263617 | 0.263617 | 1.635544 | 1.635544 | 1.635544 | 620.42 | 620.42 | 620.42 | 6.20 |
| 400 | 400 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.557934 | 0.557934 | 0.557934 | 3.334452 | 3.334452 | 3.334452 | 597.64 | 597.64 | 597.64 | 5.98 |
| 400 | 400 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 1.937759 | 1.937759 | 1.937759 | 12.005845 | 12.005845 | 12.005845 | 619.57 | 619.57 | 619.57 | 6.20 |
| 400 | 400 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 6.506604 | 6.506604 | 6.506604 | 36.085566 | 36.085566 | 36.085566 | 554.60 | 554.60 | 554.60 | 5.55 |
| 800 | 800 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.355478 | 0.355478 | 0.355478 | 2.740945 | 2.740945 | 2.740945 | 771.06 | 771.06 | 771.06 | 7.71 |
| 800 | 800 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 1.092533 | 1.092533 | 1.092533 | 7.833242 | 7.833242 | 7.833242 | 716.98 | 716.98 | 716.98 | 7.17 |
| 800 | 800 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 8.218628 | 8.218628 | 8.218628 | 48.533806 | 48.533806 | 48.533806 | 590.53 | 590.53 | 590.53 | 5.91 |
| 800 | 800 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 21.534780 | 21.534780 | 21.534780 | 115.006917 | 115.006917 | 115.006917 | 534.05 | 534.05 | 534.05 | 5.34 |
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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.261278 | 0.259515 | 0.262639 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 1.435934 | 1.406142 | 1.452845 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 5.920458 | 5.888793 | 5.944857 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 18.826368 | 18.495511 | 19.143374 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.017096 | 0.016792 | 0.017624 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.065109 | 0.064355 | 0.065737 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.252997 | 0.250580 | 0.255448 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.363573 | 0.337441 | 0.387134 |
<!-- benchmark-report:flat-arrays end -->
