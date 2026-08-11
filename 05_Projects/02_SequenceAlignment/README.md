# HPC Assignment — Reading DNA: Accelerating Smith–Waterman Sequence Alignment

This assignment is based on a compact reference implementation of the **Smith–Waterman** local alignment
algorithm (provided in `baseline/`), written in Python with NumPy — the same algorithm that sits inside
production tools such as [SSW](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library),
[Parasail](https://github.com/jeffdaily/parasail) and the alignment stage of BWA/Minimap2.

---

## Scientific Background

A DNA sequencer does not hand you a genome. It hands you hundreds of millions of short **reads**
(100–150 letters each, or 10–100 kb for long-read platforms), and every one of them has to be located
against a reference genome before anything biological can be said.

The exact method for this is **Smith–Waterman**: a dynamic-programming algorithm that fills a scoring
matrix `H` of size `(m+1) × (n+1)` for a read of length `m` and a reference segment of length `n`:

```
H[i,j] = max( 0,
              H[i-1,j-1] + s(a_i, b_j),      # match / mismatch
              H[i-1,j]   - gap,              # deletion
              H[i,j-1]   - gap )             # insertion
```

The maximum entry in `H` is the optimal local alignment score, and a traceback reconstructs the alignment
itself. The algorithm is `O(m·n)` per read pair and provably optimal — which is why it is the gold standard
and also why almost nobody can afford to run it at full scale.

- Aligning **100,000 reads** of 150 bp against a 5 Mbp bacterial reference is already ~10¹¹ cell updates.
- A straightforward Python/NumPy implementation manages on the order of a few **million cell updates per
  second (MCUPS)**. Tuned SIMD implementations reach tens of **GCUPS**; GPUs reach hundreds.
- The catch: the recurrence has a **data dependency** on `H[i-1,j-1]`, `H[i-1,j]` and `H[i,j-1]`. You cannot
  simply vectorize the inner loop. Cells on the same **anti-diagonal**, however, are independent — and
  different read/reference *pairs* are completely independent.
- A challenging extension: **affine gap penalties** (the Gotoh variant, three matrices instead of one),
  and full **traceback**, which needs either the whole matrix in memory or a divide-and-conquer
  (Hirschberg / Myers–Miller) strategy. Memory, not FLOPs, becomes the limit.

---

## Goal of the Assignment

The goal is not simply to achieve the biggest speedup.
What matters is to apply a proper engineering workflow:

- Profile and benchmark the baseline code
- Analyze the problem and formulate hypotheses
- Apply HPC techniques (see below)
- Document the journey: design decisions, experiments, results

Questions worth forming hypotheses about:

- Which parallelization axis wins: **intra-sequence** (anti-diagonal wavefront within one alignment) or
  **inter-sequence** (many alignments at once)? Under what conditions does each win?
- The score fits in an `int16` or even `int8`. How much throughput does narrowing the datatype buy you on
  a CPU vector unit — and when does saturation/overflow become a correctness problem?
- The anti-diagonal layout is cache-hostile. Does re-laying-out the matrix (striped/Farrar layout) pay off?
- Reads have different lengths. Quantify the **load imbalance** and the cost of padding vs. bucketing.
- On a GPU, is one thread per alignment better than one warp per alignment? Measure occupancy, not vibes.
- You do not need the full matrix if you only want the score. How does that change the memory footprint,
  and what does it cost you when you *do* need the traceback?

---

## Baseline and Data

The baseline is in [`baseline/`](baseline/) and needs nothing but NumPy:

| File | What it is |
|------|-----------|
| [`baseline/generate_data.py`](baseline/generate_data.py) | Generates a reference sequence and reads sampled from it with substitutions and indels, plus the true origin of every read. Reproducible from a fixed seed. |
| [`baseline/smith_waterman.py`](baseline/smith_waterman.py) | The code you optimize: Gotoh affine-gap Smith–Waterman, one Python iteration per matrix cell. |
| [`baseline/validate.py`](baseline/validate.py) | Bit-identity regression against the baseline, plus a plausibility check that alignments land where the reads came from. |

```bash
cd baseline
python generate_data.py  --size small     # tiny | small | medium | large
python smith_waterman.py --size small
python validate.py       --size small --candidate ../results/sw_gpu.tsv
```

The workload models the **realignment** stage of a read mapper: each read has already been seeded to a
candidate window of the reference, and the exact score for that read/window pair has to be computed. That
keeps the problem honest without requiring you to implement a seeding stage first.

| preset | reads | read len | window | DP cells | baseline runtime |
|--------|-------|----------|--------|----------|------------------|
| tiny   | 20    | 100      | 400    | 0.8 M    | 0.25 s (measured) |
| small  | 200   | 150      | 1,000  | 30 M     | 8.3 s (measured) |
| medium | 2,000 | 150      | 2,000  | 600 M    | ~2.8 min (estimated) |
| large  | 20,000| 250      | 4,000  | 20 G     | ~1.5 h (estimated) |

Measured on 2 vCPU with NumPy 2.2. The baseline sustains **~3.6 MCUPS**; a tuned SIMD implementation
reaches tens of **GCUPS** and a GPU hundreds, so there are four orders of magnitude on the table here.
Re-measure on `pub030` before quoting any speedup.

**Real data.** Swap in *E. coli* K-12 MG1655 (~4.6 Mbp) from
[NCBI](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3) and a real Illumina run from the
[SRA](https://www.ncbi.nlm.nih.gov/sra) once your pipeline works; human chromosome 20 (~64 Mbp) if you
want the memory pressure to become interesting. Benchmarking against
[Parasail](https://github.com/jeffdaily/parasail) or [SSW](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)
tells you how far you actually got.

**Correctness requirement:** `validate.py` must report **bit-identical scores** to the baseline. These are
integers — there is no floating-point excuse. If a narrowed datatype saturates or a heuristic changes the
answer, say so explicitly, quantify the deviation, and defend it.

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
- CPU vectorization (anti-diagonal or striped/Farrar layouts, Numba, explicit use of CPU vector units)
- Narrowed integer datatypes (`int16`/`int8`) with overflow handling
- GPU acceleration (CuPy, Numba CUDA) — thread-per-alignment vs. warp-per-alignment
- MPI (distributing read batches across processes and nodes, multi-GPU)
- Dask (task graphs over read chunks, dynamic scheduling to fight load imbalance)
- Hybrid approaches
- Benchmarking against a tuned reference such as Parasail to see how far you got

Report your results in **GCUPS** (giga cell updates per second) so they can be compared to the literature.

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
  - Benchmark results and comparisons (including GCUPS)
  - A correctness comparison against the baseline
  - Reflections on what worked well, what did not, and why

---

## Summary

This project takes one of the most-executed algorithms in the life sciences and asks you to make it fast.
Unlike an embarrassingly parallel workload, Smith–Waterman has a genuine data dependency, so you have to
*choose* a parallelization axis and defend that choice. You are expected to make use of CPU vector units,
GPUs, MPI, and Dask, applying them in a structured way. The focus is on correctness, analysis, and
engineering discipline — speedup numbers are secondary.
