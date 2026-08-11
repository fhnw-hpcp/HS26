# HPC Assignment — A Warming Europe: Accelerating Climate Index Computation on ERA5

This assignment is based on [xclim](https://github.com/Ouranosinc/xclim), an operational climate-services
library built on [xarray](https://github.com/pydata/xarray) and NumPy, applied to the
[ERA5 reanalysis](https://cds.climate.copernicus.eu/) — the most widely used climate dataset in the world.

---

## Scientific Background

**ERA5** is a reconstruction of the global atmosphere from 1940 to the present, produced by ECMWF. It is a
4-dimensional grid: latitude × longitude × time (hourly) × variable, at 0.25° resolution — roughly
**1440 × 721 grid cells**, **~750,000 hourly time steps**, and dozens of variables. The full archive is
petabytes; even a single variable over Europe at daily resolution is tens of gigabytes.

Climate scientists rarely look at raw temperature fields. They compute **climate indices**: derived,
policy-relevant quantities defined by the
[ETCCDI](https://www.climdex.org/learn/indices/) standard, such as

- **TX90p** — the fraction of days whose maximum temperature exceeds the 90th percentile of a reference
  period, computed *per calendar day* with a moving window (a percentile per grid cell per day-of-year)
- **Heat wave duration index (HWDI)** — runs of ≥ 6 consecutive days above a threshold
- **Growing degree days**, **frost days**, **consecutive dry days**, **R95pTOT** (precipitation from very wet days)

These indices are what feed into IPCC reports, insurance risk models and national adaptation plans.

The computational character is completely different from a classic simulation:

- The arithmetic per data element is **trivial** — comparisons, counts, cumulative sums.
- The dataset does not fit in memory, so this is an **out-of-core**, I/O- and memory-bandwidth-bound problem.
- Several indices (percentile-based ones, run-length ones) are **not** simple element-wise reductions.
  They need a full pass over the time axis per grid cell, which fights against the chunk layout you'd
  choose for spatial operations.
- A naive `xarray` + Dask pipeline frequently ends up **slower than serial**, or dies with a memory blow-up,
  because of bad chunking. Diagnosing that is the heart of this project.
- A challenging extension: run the same indices over a **multi-model CMIP6 ensemble** and compute
  ensemble statistics — the workload becomes genuinely distributed.

---

## Goal of the Assignment

The goal is not simply to achieve the biggest speedup.
What matters is to apply a proper engineering workflow:

- Profile and benchmark the baseline code
- Analyze the problem and formulate hypotheses
- Apply HPC techniques (see below)
- Document the journey: design decisions, experiments, results

Questions worth forming hypotheses about:

- What is the **arithmetic intensity** of TX90p? Where is it on the roofline? Is a GPU even the right tool?
- How does **chunk shape** (time-contiguous vs. space-contiguous vs. balanced) change wall time, peak memory
  and the size of the Dask graph? Find the cliff, and explain it.
- **NetCDF4/HDF5 vs. Zarr**: how much of your runtime is decompression? Does the compressor choice
  (zlib vs. zstd vs. blosc-lz4) matter more than the parallelism?
- Run-length indices like HWDI need state along the time axis. Can they be expressed as an associative
  reduction so they parallelize across chunks — or must the time axis stay in one chunk?
- Does the Dask **scheduler** (threads / processes / distributed) matter, and does the GIL actually bite here?
- When does moving to GPU arrays (`cupy-xarray`, `kvikio`) pay for the PCIe transfer, and when is it a loss?

---

## Baseline and Data

The baseline is in [`baseline/`](baseline/) and needs NumPy plus `netCDF4`:

| File | What it is |
|------|-----------|
| [`baseline/generate_data.py`](baseline/generate_data.py) | Generates an ERA5-like daily NetCDF4 file over Europe — seasonal cycle by latitude, warming trend, spatially correlated and persistent weather noise, realistic wet/dry precipitation. Chunked and compressed, so it behaves like the real thing on disk. |
| [`baseline/climate_indices.py`](baseline/climate_indices.py) | The code you optimize: TX90p, HWDI, GDD and CDD computed one grid cell at a time, with `np.percentile` called once per grid cell per day of year. |
| [`baseline/validate.py`](baseline/validate.py) | Per-index regression against the baseline, with tolerances chosen per index. |

```bash
cd baseline
python generate_data.py    --size small    # tiny | small | medium | large
python climate_indices.py  --size small
python validate.py         --size small --candidate ../results/indices_dask.npz
```

`--cells N` limits the run to the first N grid cells, which is how you get a timing signal in seconds
rather than hours while developing.

| preset | years | grid | values/variable | on disk | baseline runtime |
|--------|-------|------|-----------------|---------|------------------|
| tiny   | 5     | 12 × 16   | 0.4 M   | 3 MB   | 2.7 s (measured) |
| small  | 20    | 30 × 40   | 8.8 M   | 79 MB  | ~23 s (measured 18.9 ms/cell) |
| medium | 45    | 60 × 80   | 79 M    | ~700 MB| ~2.5 min (estimated) |
| large  | 45    | 200 × 280 | 919 M   | ~8 GB  | ~30 min (estimated) |

Measured on 2 vCPU with NumPy 2.2. **TX90p alone is 73–95 % of the runtime** and GDD is nearly free —
the spread between the four indices is the point of the exercise, so profile them separately.

Note the deliberate trap: the file is chunked along **time**, which is what a naive conversion gives you
and exactly the wrong layout for the per-cell percentile computation. Rechunking is a legitimate
optimization, but you have to count the cost of doing it.

**Real data.** ERA5 daily `tasmax`, `tasmin`, `tas` and `pr` over Europe is pre-staged on the cluster in
both NetCDF4 and Zarr; register at [Copernicus CDS](https://cds.climate.copernicus.eu/) for additional
variables. ARCO-ERA5 is worth studying as a reference Zarr layout, and
[xclim](https://github.com/Ouranosinc/xclim) gives you a production implementation to compare against.

**Correctness requirement:** `validate.py` must pass for every index. Percentile definitions are the
classic source of silent disagreement — `np.percentile`, `np.quantile` and the ETCCDI bootstrapped
definition do not all interpolate identically. Pin down the interpolation method and document it.

---

## Environment

Start your work on [pub030.cs.technik.fhnw.ch](https://pub030.cs.technik.fhnw.ch).

If you require access to the FHNW Slurm Cluster, write an Email to
[Manuel Stutz](mailto:manuel.stutz@fhnw.ch) and send your ed25519 public key to him.

> Note: To generate an ed25519 key, use the following command: `ssh-keygen -t ed25519`.

---

## Assessment and Grading

- 30% — Entry test (already fixed)
- 20% — Quality of the assignment write-up
  - Structure, clarity, and documentation
  - Well-formed hypotheses and testing against them
- 50% — Implemented speedup strategies
  - May include multiple techniques or combined approaches

---

## Group Work

You may work in groups of 2 to 4 students.
- A group of 2 must implement at least 2 different strategies.
- A group of 4 must implement at least 4 different strategies.

Possible strategies include:
- CPU vectorization (NumPy, `numpy.lib.stride_tricks`, Numba `guvectorize`, exploiting CPU vector units)
- Dask: chunking strategy, `map_blocks`, the distributed scheduler, dashboard-driven diagnosis
- Storage-format and compression engineering (NetCDF4 → Zarr, rechunking, consolidated metadata)
- GPU acceleration (CuPy, `cupy-xarray`, Numba CUDA kernels for the run-length indices)
- MPI (domain decomposition across nodes, `mpi4py`, or `dask-mpi` / `dask-jobqueue` on Slurm)
- Hybrid approaches

> Note: this project is the one where **the obvious parallelization can make things worse**. A well-argued
> negative result, properly measured, is worth full marks here.

---

## Deliverables

Submit your work either:
- As a Git repository (with access granted to `@UeliDeSchwert` and `@simonmarcin`), or
- As a zip file containing your report and code base

Your submission should include:
- The code with your optimizations
- A report (Markdown, PDF, or Notebook) documenting:
  - Profiling and analysis of the baseline (including peak memory, not just wall time)
  - Implemented strategies and rationale
  - Benchmark results and comparisons, with a scaling study
  - A correctness comparison against the baseline
  - Reflections on what worked well, what did not, and why

---

## Summary

This project is a real climate-services workload: enormous data, trivial arithmetic, and a performance
story dominated by I/O, memory layout and chunking rather than FLOPs. It is the counterweight to the
compute-bound projects — the lesson is that *knowing which resource you are actually short of* is the whole
job. You are expected to make use of CPU vector units, GPUs, MPI, and Dask, applying them in a structured
way. The focus is on correctness, analysis, and engineering discipline — speedup numbers are secondary.
