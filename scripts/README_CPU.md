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
| 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.039398 | 0.038856 | 0.039753 |
| 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.106508 | 0.102081 | 0.112669 |
| 800 | 500 | 5 | 1 | 500 | 500 | 5 | 3 | 0.359438 | 0.354758 | 0.363241 |
| 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 3 | 1.043017 | 1.032541 | 1.055330 |
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
| 2000 | 1981 | 1980 | 1980 | 10 | 3 | 0.008019 | 0.007410 | 0.009056 |
| 8000 | 7981 | 7980 | 7980 | 10 | 3 | 0.077036 | 0.076556 | 0.077811 |
| 32000 | 31981 | 31980 | 31980 | 10 | 3 | 1.240683 | 1.235914 | 1.247270 |
| 128000 | 127981 | 127980 | 127980 | 10 | 3 | 19.807366 | 19.635135 | 20.032330 |
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
| 50 | 50 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.018473 | 0.018473 | 0.018473 | 0.080324 | 0.080324 | 0.080324 | 434.83 | 434.83 | 434.83 | 4.35 |
| 50 | 50 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.034160 | 0.034160 | 0.034160 | 0.171825 | 0.171825 | 0.171825 | 503.00 | 503.00 | 503.00 | 5.03 |
| 50 | 50 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.227881 | 0.227881 | 0.227881 | 1.087089 | 1.087089 | 1.087089 | 477.04 | 477.04 | 477.04 | 4.77 |
| 50 | 50 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 0.634859 | 0.634859 | 0.634859 | 3.425090 | 3.425090 | 3.425090 | 539.50 | 539.50 | 539.50 | 5.40 |
| 100 | 100 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.019847 | 0.019847 | 0.019847 | 0.112084 | 0.112084 | 0.112084 | 564.73 | 564.73 | 564.73 | 5.65 |
| 100 | 100 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.043027 | 0.043027 | 0.043027 | 0.271998 | 0.271998 | 0.271998 | 632.15 | 632.15 | 632.15 | 6.32 |
| 100 | 100 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.362127 | 0.362127 | 0.362127 | 2.137646 | 2.137646 | 2.137646 | 590.30 | 590.30 | 590.30 | 5.90 |
| 100 | 100 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 1.262590 | 1.262590 | 1.262590 | 6.938144 | 6.938144 | 6.938144 | 549.52 | 549.52 | 549.52 | 5.50 |
| 200 | 200 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.046165 | 0.046165 | 0.046165 | 0.294361 | 0.294361 | 0.294361 | 637.63 | 637.63 | 637.63 | 6.38 |
| 200 | 200 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.121888 | 0.121888 | 0.121888 | 0.754039 | 0.754039 | 0.754039 | 618.63 | 618.63 | 618.63 | 6.19 |
| 200 | 200 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 0.776195 | 0.776195 | 0.776195 | 4.711468 | 4.711468 | 4.711468 | 607.00 | 607.00 | 607.00 | 6.07 |
| 200 | 200 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 2.796656 | 2.796656 | 2.796656 | 15.335359 | 15.335359 | 15.335359 | 548.35 | 548.35 | 548.35 | 5.48 |
| 400 | 400 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.227583 | 0.227583 | 0.227583 | 1.433243 | 1.433243 | 1.433243 | 629.77 | 629.77 | 629.77 | 6.30 |
| 400 | 400 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 0.364548 | 0.364548 | 0.364548 | 2.313510 | 2.313510 | 2.313510 | 634.62 | 634.62 | 634.62 | 6.35 |
| 400 | 400 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 1.919827 | 1.919827 | 1.919827 | 11.572907 | 11.572907 | 11.572907 | 602.81 | 602.81 | 602.81 | 6.03 |
| 400 | 400 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 6.354642 | 6.354642 | 6.354642 | 34.680984 | 34.680984 | 34.680984 | 545.76 | 545.76 | 545.76 | 5.46 |
| 800 | 800 | 500 | 5 | 1 | 500 | 500 | 5 | 1 | 0.339471 | 0.339471 | 0.339471 | 2.601845 | 2.601845 | 2.601845 | 766.44 | 766.44 | 766.44 | 7.66 |
| 800 | 800 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 1 | 1.019268 | 1.019268 | 1.019268 | 6.821562 | 6.821562 | 6.821562 | 669.26 | 669.26 | 669.26 | 6.69 |
| 800 | 800 | 4000 | 5 | 1 | 4000 | 4000 | 5 | 1 | 6.995436 | 6.995436 | 6.995436 | 37.464097 | 37.464097 | 37.464097 | 535.55 | 535.55 | 535.55 | 5.36 |
| 800 | 800 | 8000 | 5 | 1 | 8000 | 8000 | 5 | 1 | 21.512179 | 21.512179 | 21.512179 | 92.803518 | 92.803518 | 92.803518 | 431.40 | 431.40 | 431.40 | 4.31 |
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
| 1000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.253263 | 0.245604 | 0.266753 |
| 5000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 1.339994 | 1.328126 | 1.356146 |
| 20000 | 1 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 5.499268 | 5.452363 | 5.530051 |
| 20000 | 1 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 18.562382 | 18.535557 | 18.614591 |
| 1 | 1000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.014881 | 0.013973 | 0.015391 |
| 1 | 5000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.064961 | 0.063780 | 0.065844 |
| 1 | 20000 | 1000 | 5 | 1 | 1000 | 1000 | 5 | 3 | 0.260316 | 0.255178 | 0.266061 |
| 1 | 20000 | 2000 | 5 | 1 | 2000 | 2000 | 5 | 3 | 0.360033 | 0.346673 | 0.373669 |
<!-- benchmark-report:flat-arrays end -->
