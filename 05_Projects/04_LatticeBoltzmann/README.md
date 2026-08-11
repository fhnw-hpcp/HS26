# HPC Assignment — Simulating Fluid Flow: Accelerating a Lattice Boltzmann Solver

This assignment is based on a compact **Lattice Boltzmann Method (LBM)** fluid solver (provided in
`baseline/`), a ~150-line NumPy implementation in the tradition of Jonas Latt's classic Python LBM and the
[Palabos](https://palabos.unige.ch/) reference cases.

---

## Scientific Background

The Lattice Boltzmann Method solves fluid dynamics not by discretizing the Navier–Stokes equations directly,
but by tracking **particle distribution functions** `f_i(x, t)` on a regular lattice. Each cell holds 9
(D2Q9) or 19/27 (D3Q19/D3Q27) numbers, one per discrete velocity direction. Every timestep is two steps:

1. **Collide** — relax the distributions locally toward equilibrium (pure local arithmetic)
2. **Stream** — shift each `f_i` to the neighbouring cell in direction `i` (pure data movement)

plus boundary conditions (bounce-back at walls, inlet/outlet). Macroscopic density and velocity are simple
moments of `f`. It is elegant, second-order accurate, handles complex geometry naturally, and it is
*everywhere* — from blood flow in aneurysms to airflow around cyclists to porous-media simulation.

The classic benchmark case, and the one you will run, is **flow past a cylinder** at Reynolds number
~200, where the simulation develops a von Kármán vortex street.

- A useful 2D domain is **4000 × 1000 cells** run for 100,000 timesteps; 3D turns that into
  **512³ cells** and 19 distributions per cell — several GB of state.
- The plain NumPy baseline manages a few **MLUPS** (mega lattice updates per second). Optimized CPU codes
  reach hundreds of MLUPS per socket; GPUs reach tens of **GLUPS**.
- LBM is the textbook **memory-bandwidth-bound stencil**: it touches ~2 × 9 × 8 bytes per cell update for
  a handful of FLOPs. Arithmetic intensity is low and essentially fixed — which makes it a perfect subject
  for a **roofline analysis**.
- The streaming step is the interesting part: it is a shift, so a naive implementation allocates a full
  second copy of the lattice every timestep. Whether you can avoid that (A-B vs. A-A pattern, push vs. pull,
  esoteric twist) determines your bandwidth ceiling.
- A challenging extension: **multi-node / multi-GPU domain decomposition** with halo exchange, and
  overlapping communication with computation.

---

## Goal of the Assignment

The goal is not simply to achieve the biggest speedup.
What matters is to apply a proper engineering workflow:

- Profile and benchmark the baseline code
- Analyze the problem and formulate hypotheses
- Apply HPC techniques (see below)
- Document the journey: design decisions, experiments, results

Questions worth forming hypotheses about:

- Measure the achieved bandwidth and place your kernel on a **roofline**. What fraction of STREAM
  bandwidth do you reach, and what is stopping you from reaching more?
- **Data layout**: array-of-structures (`f[x,y,i]`) vs. structure-of-arrays (`f[i,x,y]`). Predict which is
  faster and by how much *before* you measure — then explain the gap.
- Can you **fuse** collide and stream into a single pass? What does that do to memory traffic, and what
  does it cost in code clarity and boundary-condition complexity?
- How much does the extra lattice copy cost? Try an in-place scheme and quantify the difference.
- On a GPU, how do coalescing and the streaming shift interact? Is a shared-memory tile worth it for a
  stencil this shallow?
- With MPI domain decomposition: at what domain size does halo exchange stop being negligible? Does
  non-blocking communication with overlapped interior computation recover it? Show a strong *and* a weak
  scaling study.

---

## Baseline and Data

The baseline is in [`baseline/`](baseline/) and needs nothing but NumPy:

| File | What it is |
|------|-----------|
| [`baseline/lbm_d2q9.py`](baseline/lbm_d2q9.py) | The code you optimize: D2Q9, BGK collision, bounce-back walls, `np.roll` streaming into a fresh array. Two cases — flow past a cylinder, and lid-driven cavity. |
| [`baseline/validate.py`](baseline/validate.py) | Physics checks (mass conservation, no-slip, Mach limit) plus field-level regression against a reference run. |

```bash
cd baseline
python lbm_d2q9.py --case cylinder --size small
python lbm_d2q9.py --case cavity   --size small --out ../data/ref_cavity.npz
python validate.py --baseline ../data/ref_cavity.npz --candidate ../results/cavity_gpu.npz
```

**No external dataset needed** — initial and boundary conditions are generated. This is the one project
where you control the problem size exactly, which makes clean strong- and weak-scaling studies easy.

| preset | grid | steps | lattice updates | baseline runtime |
|--------|------|-------|-----------------|------------------|
| tiny   | 100 × 40   | 500     | 2 M    | 0.22 s (measured) |
| small  | 400 × 100  | 2,000   | 80 M   | 7.9 s (measured) |
| medium | 1000 × 250 | 10,000  | 2.5 G  | ~3.5 min (estimated) |
| large  | 4000 × 1000| 100,000 | 400 G  | ~9 h (estimated) |

Measured on 2 vCPU with NumPy 2.2, at **~10–12 MLUPS**. Optimized CPU codes reach hundreds of MLUPS per
socket and GPUs reach tens of GLUPS. Always report MLUPS alongside the achieved memory bandwidth.

**Validation.** The cylinder case at Re = 200 sheds a von Kármán street with Strouhal number ~0.2; the
baseline measures **0.23** on a 240 × 60 grid over 20,000 steps, which is the shortest run that resolves
enough shedding periods to be meaningful. Shorter runs will report nonsense — that is a sampling
limitation, not a bug.

For the lid-driven cavity, the literature reference is **Ghia, Ghia & Shin (1982), *J. Comput. Phys.* 48,
387, Table I**. Transcribing those centreline values and comparing your profile against them is part of
the assignment; `validate.py` deliberately does not do it for you.

> Known baseline artifact: the crude zero-gradient outflow on the cylinder case lets mean density drift by
> a couple of per cent over a long run. `validate.py` tolerates up to 5 %. Do not make it worse; fixing it
> is optional and, if you do, say so.

**Correctness requirement:** the physics must survive your optimizations. Report the cavity centreline
profiles against Ghia et al., and state the tolerance you accept. If you switch to single precision, show
what it does to the profiles — it may genuinely be fine here, but you have to demonstrate it rather than
assert it.

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
- CPU vectorization and cache blocking (NumPy, Numba `@njit(parallel=True)`, exploiting CPU vector units)
- Data-layout engineering (AoS vs. SoA, padding, alignment) and kernel fusion
- GPU acceleration (CuPy, Numba CUDA, custom kernels with coalesced access)
- MPI domain decomposition with halo exchange; non-blocking comms overlapped with interior updates
- Multi-GPU (MPI + CUDA-aware transfers, or Dask + CuPy)
- Hybrid approaches

Report your results in **MLUPS/GLUPS** (lattice updates per second) so they can be compared to the
literature, and always alongside the achieved memory bandwidth.

---

## Deliverables

Submit your work either:
- As a Git repository (with access granted to `@UeliDeSchwert` and `@simonmarcin`), or
- As a zip file containing your report and code base

Your submission should include:
- The code with your optimizations
- A report (Markdown, PDF, or Notebook) documenting:
  - Profiling and analysis of the baseline, including a roofline model
  - Implemented strategies and rationale
  - Benchmark results, with strong and weak scaling studies
  - Physics validation against the reference cases
  - Reflections on what worked well, what did not, and why

---

## Summary

This project is the classic HPC workload: a memory-bandwidth-bound stencil that you optimize by
understanding data movement, not arithmetic. Because you own the problem size, it is also the best of the
five for rigorous scaling studies and roofline analysis. You are expected to make use of CPU vector units,
GPUs, MPI, and Dask, applying them in a structured way. The focus is on correctness, analysis, and
engineering discipline — speedup numbers are secondary.
