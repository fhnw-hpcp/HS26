# HPC Assignment — Hunting Exoplanets: Accelerating a TESS Transit Search

This assignment is based on the [Lightkurve](https://github.com/lightkurve/lightkurve) package and the
[Box Least Squares (BLS)](https://docs.astropy.org/en/stable/timeseries/bls.html) periodogram in Astropy —
a scientific application written in Python with heavy use of NumPy.

---

## Scientific Background

NASA's **Transiting Exoplanet Survey Satellite (TESS)** stares at a patch of sky for ~27 days at a time and
measures the brightness of hundreds of thousands of stars. When a planet passes in front of its host star,
the measured flux dips by a fraction of a percent for a few hours — and it does so *periodically*.

Finding those planets means, for every star, searching a large grid of trial parameters:

- trial **period** `P` (0.5 – 15 days, tens of thousands of trial values)
- trial transit **duration** `d` (a handful of values per period)
- trial transit **phase** `t0` (every phase bin of the folded light curve)

For each `(P, d, t0)` the light curve is phase-folded and a box model is fitted; the best-fitting
combination gives the *signal detection efficiency* (SDE). This is the BLS algorithm, and it is the
workhorse of essentially every transit survey.

- One TESS sector delivers roughly **20,000 two-minute-cadence light curves**, each with ~18,000 points.
- The full 2-minute target list across all sectors runs into the **hundreds of thousands** of light curves.
- Running Astropy's `BoxLeastSquares` over a fine period grid takes **seconds to minutes per star** —
  a full sector is hours to days on a single core.
- The problem is embarrassingly parallel across stars, and the inner grid search is a dense, regular
  computation that maps well onto vector units and GPUs.
- A challenging extension: **injection-recovery testing** — inject synthetic planets with known parameters
  into real light curves and measure the detection completeness of your pipeline. This multiplies the
  workload by another factor of 100–1000 and is exactly what real survey teams do.

---

## Goal of the Assignment

The goal is not simply to achieve the biggest speedup.
What matters is to apply a proper engineering workflow:

- Profile and benchmark the baseline code
- Analyze the problem and formulate hypotheses
- Apply HPC techniques (see below)
- Document the journey: design decisions, experiments, results

Questions worth forming hypotheses about:

- Where does the time actually go — the period loop, the phase folding, the binning, or FITS I/O?
- Is the inner loop compute-bound or memory-bound? What does a roofline analysis say?
- Phase folding is effectively a **scatter/histogram** operation. How does that behave on a GPU where
  many threads write to the same bin? Atomics, privatization, or sort-then-segment-reduce?
- Light curves differ in length and in the number of valid points. How much **load imbalance** does that
  cause when you distribute stars across ranks, and does dynamic scheduling fix it?
- Does single precision change the recovered planet parameters? Verify — don't assume.

---

## Baseline and Data

The baseline is in [`baseline/`](baseline/) and needs nothing but NumPy:

| File | What it is |
|------|-----------|
| [`baseline/generate_data.py`](baseline/generate_data.py) | Generates TESS-like light curves with planets injected into a known subset. Reproducible from a fixed seed — no download needed. |
| [`baseline/bls_search.py`](baseline/bls_search.py) | The code you optimize: running-median detrending, then a BLS periodogram whose block search is a plain Python double loop. |
| [`baseline/validate.py`](baseline/validate.py) | Recovery check against the injected truth, and regression check against the baseline results. |

```bash
cd baseline
python generate_data.py --size small     # tiny | small | medium | large
python bls_search.py    --size small
python validate.py      --size small
python validate.py      --size small --candidate ../results/bls_gpu.csv
```

**Scale presets** — `tiny` is for debugging, `small` for the development loop, `medium` for reporting,
`large` for the final run.

| preset | stars | points/star | trial periods | baseline runtime |
|--------|-------|-------------|---------------|------------------|
| tiny   | 8     | 1,410       | 400           | 5.6 s (measured) |
| small  | 40    | 3,760       | 1,200         | ~87 s (measured 2.17 s/star) |
| medium | 400   | 9,306       | 3,000         | ~2 h (estimated) |
| large  | 4,000 | 18,518      | 8,000         | ~10 days (estimated) |

Measured on 2 vCPU with NumPy 2.2; treat them as a starting point and re-measure on `pub030`. The
baseline sustains **~2.6 M block evaluations/s**, and the split is roughly **97 % BLS search, 3 %
detrending** — but confirm that yourself rather than taking it from this table.

**Real data.** Once your pipeline works, swap in real TESS 2-minute light curves from the
[MAST archive](https://archive.stsci.edu/missions-and-data/tess) via `lightkurve.search_lightcurve()`,
with the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) confirmed-planet table as
ground truth. A pre-staged sector is available on the cluster — please do **not** each download the
archive separately. Comparing your result against `astropy.timeseries.BoxLeastSquares` is a good
independent check of the baseline itself.

**Correctness requirement:** `validate.py` must report the same recovered planets, and the regression
check must show every star's best-fit period agreeing with the baseline within a documented tolerance
(the script defaults to 2 %). State the tolerance you accept and justify it.

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
- CPU vectorization (NumPy, Numba, exploiting CPU vector units)
- GPU acceleration (CuPy, Numba CUDA) — one star per block, or batching many stars per kernel launch
- MPI (multi-process, multi-GPU) with static vs. dynamic work distribution
- Dask (task graphs over thousands of light curves, distributed scheduler)
- Hybrid approaches
- I/O optimization: FITS vs. Parquet/Zarr, caching, avoiding the many-small-files problem

---

## Deliverables

Submit your work either:
- As a Git repository (with access granted to `@UeliDeSchwert` and `@simonmarcin`), or
- As a zip file containing your report and code base

Your submission should include:
- The code with your optimizations
- A report (Markdown, PDF, or Notebook) documenting:
  - Profiling and analysis of the baseline
  - Implemented strategies and rationale
  - Benchmark results and comparisons
  - A correctness comparison against the baseline
  - Reflections on what worked well, what did not, and why

---

## Summary

This project combines a real astronomical survey pipeline with HPC techniques.
The workload is a dense grid search wrapped in an embarrassingly parallel outer loop — a pattern that
appears everywhere in science. You are expected to make use of CPU vector units, GPUs, MPI, and Dask,
applying them in a structured way. The focus is on correctness, analysis, and engineering discipline —
speedup numbers are secondary.
