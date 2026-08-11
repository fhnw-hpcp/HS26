# HPC Assignment — Watching Proteins Move: Accelerating MD Trajectory Analysis

This assignment is based on [MDAnalysis](https://github.com/MDAnalysis/mdanalysis), a widely used Python
library for analyzing molecular dynamics simulations, with heavy use of NumPy.

---

## Scientific Background

A molecular dynamics simulation integrates Newton's equations for every atom in a system — a protein, a
membrane, a solvated drug candidate — and writes out the coordinates every few picoseconds. Running the
simulation is a solved HPC problem (GROMACS, NAMD and OpenMM are all heavily optimized). **Analyzing the
output is not.**

A typical production trajectory is:

- **100,000 – 1,000,000 atoms** (protein plus water plus ions)
- **10,000 – 1,000,000 frames**
- **tens to hundreds of gigabytes** on disk, in XTC/DCD/TRR format

and the analyses run over it are things like:

- **Radial distribution function `g(r)`** — a histogram over all pairwise distances, `O(N²)` per frame if
  done naively, and the naive version is what most people write
- **Hydrogen-bond analysis** — geometric criteria over donor/acceptor pairs, per frame, with
  periodic boundary conditions
- **Contact maps / native contacts `Q(t)`**, **RMSD/RMSF** after optimal superposition (Kabsch algorithm),
  **solvent-accessible surface area**

MDAnalysis deliberately streams the trajectory **one frame at a time** to keep memory bounded, which makes
the analysis loop look like:

```python
for ts in u.trajectory:     # <- I/O, decompression, one frame at a time
    result[ts.frame] = analyse(u.atoms.positions)   # <- O(N) to O(N^2) compute
```

This is a genuinely mixed workload — part streaming I/O, part neighbour search, part dense arithmetic —
and which part dominates depends entirely on the analysis and the system size. Finding out *which* is the
first thing you have to do.

- A challenging extension: **time-correlation functions** (e.g. hydrogen-bond lifetime autocorrelation),
  which couple frames together and break the trivially-parallel-over-frames assumption.

---

## Goal of the Assignment

The goal is not simply to achieve the biggest speedup.
What matters is to apply a proper engineering workflow:

- Profile and benchmark the baseline code
- Analyze the problem and formulate hypotheses
- Apply HPC techniques (see below)
- Document the journey: design decisions, experiments, results

Questions worth forming hypotheses about:

- Split the baseline runtime into **I/O + decompression**, **neighbour search** and **arithmetic**.
  Which dominates, and does the answer change with system size? (It should — show the crossover.)
- Replacing the `O(N²)` distance loop with a **cell list / linked-cell grid** changes the complexity to
  `O(N)`. At what N does it actually start winning, given the constant factors?
- Parallelizing over frames is the obvious move — but each worker then re-opens and re-seeks the
  trajectory. How does that interact with the compressed XTC format, which is not randomly seekable
  without an index? Is converting to a chunked format (HDF5/H5MD, Zarr) a legitimate optimization or
  cheating? (It is legitimate — but you must count the conversion cost.)
- Periodic boundary conditions add a minimum-image calculation to every distance. Can you hoist it,
  vectorize it, or fold it into the cell-list construction?
- On a GPU: pairwise distances are the ideal kernel, but histogramming them is a scatter with atomics.
  Where does the win go?
- Does the **split-apply-combine** scheme (MDAnalysis's `ParallelAnalysisBase`) scale linearly, and if not,
  where does it stop?

---

## Baseline and Data

The baseline is in [`baseline/`](baseline/) and needs nothing but NumPy:

| File | What it is |
|------|-----------|
| [`baseline/generate_data.py`](baseline/generate_data.py) | Generates the trajectory: a bonded "protein" in three-site "water", evolved by overdamped soft-sphere relaxation. Reproducible from a fixed seed. |
| [`baseline/analysis.py`](baseline/analysis.py) | The code you optimize: RDF, hydrogen bonds and contact map / Q(t), streamed one frame at a time, all-pairs. |
| [`baseline/validate.py`](baseline/validate.py) | Structural sanity checks plus exact regression on the RDF histogram, hydrogen-bond counts and Q(t). |

```bash
cd baseline
python generate_data.py --size small     # tiny | small | medium | large
python analysis.py      --size small
python analysis.py      --size small --analyses rdf --frames 20   # isolate one kernel
python validate.py      --size small --candidate ../results/analysis_gpu.npz
```

The generated trajectory is **synthetic** and says so in its docstring — the dynamics are not physically
meaningful. What matters is that the structural features the analyses look for are correct by
construction: water oxygens have a first-neighbour shell at ~0.28 nm, one hydrogen per water is aimed at
its nearest oxygen so the angle criterion actually fires, and the protein is restrained to its fold so
Q(t) stays above ~0.95. What is being benchmarked is the analysis, not the dynamics.

| preset | atoms | frames | trajectory | baseline runtime |
|--------|-------|--------|------------|------------------|
| tiny   | 1,260   | 20    | 0.3 MB  | 1.1 s (measured, 0.054 s/frame) |
| small  | 6,300   | 100   | 7.6 MB  | ~44 s (measured, 0.44 s/frame) |
| medium | 24,900  | 400   | 120 MB  | ~45 min (estimated) |
| large  | 123,000 | 2,000 | 3.0 GB  | ~day (estimated) |

Measured on 2 vCPU with NumPy 2.2. The split at `small` is **~77 % hydrogen bonds, ~22 % RDF, ~1 % contact
map, and I/O below 1 %** — which is the first interesting result of the project, because it means the
streaming-I/O story people expect to dominate does not, *at this size*. Find the size where that changes.

**Real data.** Once your pipeline works, swap in a real trajectory via MDAnalysis — public sets are at
[MDAnalysisData](https://www.mdanalysis.org/MDAnalysisData/) and
[mdshare](https://markovmodel.github.io/mdshare/), and a production trajectory is pre-staged on the
cluster. That is also when compressed XTC and its lack of random seekability become your problem rather
than a footnote. MDAnalysis's own `InterRDF` and `HydrogenBondAnalysis` are the production implementations
to compare against.

Scale along atoms *or* frames — test both axes separately, because they stress different parts of the
machine.

**Correctness requirement:** `validate.py` must pass. The RDF histogram and hydrogen-bond counts are
integer counts and must match **exactly**; Q(t) is a ratio of integer counts and must match to round-off.
None of these has a legitimate excuse to drift. Periodic-boundary handling and histogram bin edges are
where silent disagreement creeps in.

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
- Algorithmic improvement: cell lists / neighbour grids / KD-trees instead of all-pairs
- CPU vectorization (NumPy, Numba, exploiting CPU vector units)
- GPU acceleration (CuPy, Numba CUDA — batched pairwise distances and atomic histogramming)
- MPI (frame-range decomposition across ranks, parallel trajectory readers)
- Dask (split-apply-combine over frame blocks, distributed scheduler)
- I/O and storage-format engineering (XTC → H5MD/Zarr, chunking, compression trade-offs)
- Hybrid approaches

---

## Deliverables

Submit your work either:
- As a Git repository (with access granted to `@UeliDeSchwert` and `@simonmarcin`), or
- As a zip file containing your report and code base

Your submission should include:
- The code with your optimizations
- A report (Markdown, PDF, or Notebook) documenting:
  - Profiling and analysis of the baseline, with the runtime split into I/O vs. compute
  - Implemented strategies and rationale
  - Benchmark results and scaling in both atoms and frames
  - A correctness comparison against the baseline
  - Reflections on what worked well, what did not, and why

---

## Summary

This project sits at the boundary between an I/O problem and a compute problem, and the honest answer is
"it depends on the analysis" — which is exactly what makes it a good HPC exercise. You will have to measure
before you optimize, and the biggest single win available is algorithmic rather than hardware. You are
expected to make use of CPU vector units, GPUs, MPI, and Dask, applying them in a structured way. The focus
is on correctness, analysis, and engineering discipline — speedup numbers are secondary.
