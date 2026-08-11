# HPC Summer School — Assignment Projects

Five projects, each a real scientific workload with a slow Python/NumPy baseline. Every project follows the
same structure and the same grading scheme; they differ in **which resource you run out of first**, which is
the actual lesson.

Groups pick **one** project. Groups of 2 implement at least 2 strategies, groups of 4 at least 4.

---

## The projects

| # | Project | Domain | Baseline | Dominant bottleneck | Where the win comes from |
|---|---------|--------|----------|---------------------|--------------------------|
| [01](01_TransitSearch/) | **Hunting Exoplanets** — BLS transit search in TESS light curves | Astronomy | Real: [Lightkurve](https://github.com/lightkurve/lightkurve) + Astropy `BoxLeastSquares` | Compute-bound grid search; load imbalance; many small files | Task parallelism, GPU batching, scatter/histogram on GPU |
| [02](02_SequenceAlignment/) | **Reading DNA** — Smith–Waterman local alignment | Bioinformatics | Provided NumPy implementation | Compute-bound with a real data dependency | SIMD/wavefront, narrow integer types, choosing the parallel axis |
| [03](03_ClimateIndices/) | **A Warming Europe** — ETCCDI climate indices on ERA5 | Climate / geospatial | Real: [xclim](https://github.com/Ouranosinc/xclim) + xarray | I/O, decompression, chunking, memory | Chunk shape, storage format, Dask done properly |
| [04](04_LatticeBoltzmann/) | **Simulating Fluid Flow** — Lattice Boltzmann solver | CFD | Provided NumPy implementation | Memory bandwidth (classic stencil) | Data layout, kernel fusion, halo exchange, roofline |
| [05](05_MDTrajectory/) | **Watching Proteins Move** — MD trajectory analysis | Molecular dynamics | Real: [MDAnalysis](https://github.com/MDAnalysis/mdanalysis) | Mixed: streaming I/O + O(N²) neighbour search | Algorithmic (cell lists) first, then hardware |

Every project ships a working baseline in its `baseline/` folder: a data generator, the slow
implementation to optimize, and a validator. All five run on **NumPy alone** (project 03 also needs
`netCDF4`). Nothing has to be downloaded to get started.

```bash
pip install -r requirements.txt
cd 04_LatticeBoltzmann/baseline && python lbm_d2q9.py --size small
```

Each baseline has `tiny` / `small` / `medium` / `large` presets — `tiny` for debugging, `small` for the
development loop, `medium` for reporting, `large` for the final run. The per-project READMEs give measured
baseline runtimes for each.

> Generated data goes in `<project>/data/` and is gitignored. The large presets run to several GB —
> do not commit them.

---

## Choosing

- **01 (Transit search)** is the most approachable — embarrassingly parallel outer loop, quick early wins,
  with depth available in the GPU inner kernel and injection-recovery testing.
- **02 (Smith–Waterman)** is the best fit for a group interested in low-level CPU vectorization; it is the
  only project where the algorithm actively resists naive parallelization.
- **03 (Climate indices)** is the only project where the obvious parallelization frequently makes things
  slower. Well suited to a group willing to write up a rigorous negative result.
- **04 (Lattice Boltzmann)** is the cleanest for scaling studies and roofline analysis — you control the
  problem size exactly, and there is no data to download. Good default for a group that wants to go deep on
  MPI and GPU kernels.
- **05 (MD trajectories)** rewards groups who profile honestly before optimizing; the largest single win is
  algorithmic, not hardware.

---

## Common to all projects

**Grading**

- 30% — Entry test (already fixed)
- 20% — Quality of the assignment write-up
- 50% — Implemented speedup strategies

**Environment** — start on [pub030.cs.technik.fhnw.ch](https://pub030.cs.technik.fhnw.ch). For FHNW Slurm
cluster access, email [Manuel Stutz](mailto:manuel.stutz@fhnw.ch) with your ed25519 public key
(`ssh-keygen -t ed25519`).

**Submission** — a Git repository with access for `@UeliDeSchwert` and `@simonmarcin`, or a zip file, containing
the optimized code plus a report covering baseline profiling, strategies and rationale, benchmarks,
a correctness comparison against the baseline, and reflections.

**Non-negotiable** — every project defines a correctness requirement against the baseline, checked by its
`validate.py`. A fast wrong answer scores zero. Speedup numbers are secondary to correctness, analysis,
and engineering discipline.

---

## Baseline performance at a glance

Measured on 2 vCPU with NumPy 2.2 at the `small` preset. These exist so you can tell whether your machine
is behaving, not as a target — re-measure on `pub030` before quoting any speedup.

| # | Project | `small` runtime | Native throughput unit | Dominant cost |
|---|---------|-----------------|------------------------|---------------|
| 01 | Transit search | ~87 s | 2.6 M block evals/s | 97 % BLS block search |
| 02 | Smith–Waterman | 8.3 s | 3.6 MCUPS | ~100 % DP cell loop |
| 03 | Climate indices | ~23 s | 18.9 ms per grid cell | 73–95 % TX90p percentiles |
| 04 | Lattice Boltzmann | 7.9 s | 10–12 MLUPS | streaming + collision, bandwidth-bound |
| 05 | MD analysis | ~44 s | 0.44 s per frame | 77 % hydrogen bonds, 22 % RDF, <1 % I/O |
