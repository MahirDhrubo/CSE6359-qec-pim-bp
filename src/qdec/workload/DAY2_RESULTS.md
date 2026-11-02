# P1 Day 2 TEST comparison

Selected configuration: 6-bit messages, range x1.5. For legacy traces without metadata, range provenance is not certified.

Float = our min-sum BP + our OSD; CPU timing baseline = ldpc BP+OSD.

PASS means no significant degradation detected, not proven equivalence. All intervals are 95% Wilson; p-values are one-sided paired McNemar. TEST results were inspected during earlier iterations; this is not a blinded study.

| Workload | Bits | N | Quant failures / LER [CI] | Float failures / LER [CI] | Matching failures | Only quant / only float | p | LER screen |
|---|---:|---:|---|---|---:|---:|---:|---|
| surface_d5_p0.001_s2000 | 6 | 20000 | 4 / 0.0002 [7.77775e-05, 0.000514189] | 4 / 0.0002 [7.77775e-05, 0.000514189] | 4 | 0 / 0 | 1 | PASS |
| surface_d5_p0.003_s2000 | 6 | 20000 | 36 / 0.0018 [0.00130052, 0.00249083] | 37 / 0.00185 [0.00134253, 0.0025488] | 67 | 6 / 7 | 0.709473 | PASS |
| surface_d7_p0.001_s2000 | 6 | 20000 | 0 / 0 [0, 0.000192043] | 0 / 0 [0, 0.000192043] | 1 | 0 / 0 | 1 | PASS |
| surface_d7_p0.003_s2000 | 6 | 20000 | 12 / 0.0006 [0.000343267, 0.00104855] | 12 / 0.0006 [0.000343267, 0.00104855] | 28 | 5 / 5 | 0.623047 | PASS |

| Workload | Quantized OSD rate | Float OSD rate | OSD ratio | Quantized mean iterations | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|
| surface_d5_p0.001_s2000 | 12.905% | 11.680% | 1.1049 | 7.3096 | 50 | 50 |
| surface_d5_p0.003_s2000 | 42.750% | 38.250% | 1.1176 | 22.5358 | 50 | 50 |
| surface_d7_p0.001_s2000 | 26.665% | 24.595% | 1.0842 | 14.7985 | 50 | 50 |
| surface_d7_p0.003_s2000 | 73.360% | 69.060% | 1.0623 | 37.6688 | 50 | 50 |

Float OSD rates come only from verified generation sidecars; legacy traces without those diagnostics report unavailable.

Inspected artifact SHA-256 (does not certify original generation settings):

- `data/traces/functional_surface_d5_p0.001_s2000.parquet`: `3c989b8a6ae81f88cceab64aca0f5f7d7490a146f15373f40b6498fe5cc1e78c`
- `data/shots/surface_d5_p0.001_s2000.npz`: `6ea4987e343505e349468c8438e7ec3247a5df88d8b0600474b99dae2d68ba64`
- `data/traces/functional_surface_d5_p0.003_s2000.parquet`: `80a2c79db23f54d1c76a0cff7e42ad7a7a360602d204da5a710e32aa3ca8151a`
- `data/shots/surface_d5_p0.003_s2000.npz`: `dc40571468b69b0e220cce2321de3d50e6e7c3b31f964bf44f7c5f0ab5de3ee0`
- `data/traces/functional_surface_d7_p0.001_s2000.parquet`: `74052a3388df5336b8af1dd4157402a224a8a3e77df5356f63b4fda9e01eb8e5`
- `data/shots/surface_d7_p0.001_s2000.npz`: `2d0207947923cd7451d4c0b0d16d1e666361213443e013396df460434c8ac733`
- `data/traces/functional_surface_d7_p0.003_s2000.parquet`: `cebd9b6b03fa279251ea59c56b28e04e94b3ca15edccd99e166a24d2387d9a97`
- `data/shots/surface_d7_p0.003_s2000.npz`: `3ecd26a4ddb06be90357b9e1f9922bd9b64656baa67681aa74c378a4aef1b476`
