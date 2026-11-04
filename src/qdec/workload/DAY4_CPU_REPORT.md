# P1 CPU baseline: complete reduced profile

All 23 workloads, 1,000 TEST shots each. MEASURED whole ldpc BP+OSD decode latency in milliseconds, with one decoder thread. These are not OSD-only costs or custom Python BP+OSD timings.

Machine/software metadata is in each JSON sidecar. Timing includes ordinary host activity and overlap with pre-commit tests; see measurement_activity.log in the final package. At 1,000 shots p99.9 is near the maximum and is not a well-resolved tail.

| Tag | N | p50 ms | p99 ms | p99.9 ms | max ms |
|---|---:|---:|---:|---:|---:|
| gross_d12_p0.001_s2000 | 1000 | 13633.8 | 17296.6 | 18233.7 | 19058.2 |
| gross_d12_p0.002_s2000 | 1000 | 14489.3 | 19638.9 | 20808.6 | 21344.4 |
| gross_d12_p0.003_s2000 | 1000 | 15739.7 | 21838 | 23905.2 | 24287.7 |
| surface_d11_p0.001_s2000 | 1000 | 861.142 | 897.492 | 947.055 | 1555.85 |
| surface_d11_p0.002_s2000 | 1000 | 885.746 | 1029.86 | 1656.04 | 2531.17 |
| surface_d11_p0.003_s2000 | 1000 | 899.386 | 1001.32 | 1742.49 | 1848.54 |
| surface_d11_p0.005_s2000 | 1000 | 927.262 | 1048.64 | 1251.18 | 1759.94 |
| surface_d13_p0.001_s2000 | 1000 | 2494 | 2861.97 | 3332.02 | 3362.45 |
| surface_d13_p0.002_s2000 | 1000 | 2551.86 | 2774.87 | 3544.06 | 3613.06 |
| surface_d13_p0.003_s2000 | 1000 | 2605.91 | 2838.15 | 4737.83 | 6684.53 |
| surface_d13_p0.005_s2000 | 1000 | 2684.67 | 3059.97 | 5359.95 | 5387.82 |
| surface_d5_p0.001_s2000 | 1000 | 0.0886665 | 5.43476 | 5.50835 | 5.52833 |
| surface_d5_p0.002_s2000 | 1000 | 0.117062 | 5.578 | 5.6726 | 5.68929 |
| surface_d5_p0.003_s2000 | 1000 | 0.146521 | 5.83154 | 6.05117 | 6.13971 |
| surface_d5_p0.005_s2000 | 1000 | 5.37181 | 5.74503 | 6.16121 | 6.20342 |
| surface_d7_p0.001_s2000 | 1000 | 0.427229 | 47.7761 | 48.933 | 73.2545 |
| surface_d7_p0.002_s2000 | 1000 | 1.80119 | 49.3293 | 52.1164 | 53.8345 |
| surface_d7_p0.003_s2000 | 1000 | 47.4038 | 53.3196 | 78.862 | 86.0734 |
| surface_d7_p0.005_s2000 | 1000 | 48.7982 | 52.7358 | 89.3288 | 150.443 |
| surface_d9_p0.001_s2000 | 1000 | 1.4006 | 247.668 | 317.125 | 317.141 |
| surface_d9_p0.002_s2000 | 1000 | 242.012 | 253.279 | 270.99 | 272.04 |
| surface_d9_p0.003_s2000 | 1000 | 246.972 | 257.196 | 390.521 | 823.212 |
| surface_d9_p0.005_s2000 | 1000 | 252.797 | 264.909 | 279.907 | 286.679 |
