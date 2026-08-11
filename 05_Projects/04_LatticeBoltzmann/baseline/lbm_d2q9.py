#!/usr/bin/env python3
"""
Baseline D2Q9 Lattice Boltzmann solver — deliberately naive, correct, and slow.

This is the code you are asked to make fast. It is written the way a
computational scientist writes it on a first pass: readable NumPy, one array
operation per lattice direction, and a fresh array allocated wherever that was
the obvious thing to do.

Two cases are provided:
  cylinder  -- flow past a cylinder, develops a von Karman vortex street
  cavity    -- lid-driven cavity, the standard validation case

Usage
-----
    python lbm_d2q9.py --case cylinder --size small
    python lbm_d2q9.py --case cavity --size medium --out results/cavity.npz

Performance is reported in MLUPS (mega lattice updates per second):
    MLUPS = nx * ny * nsteps / runtime / 1e6

Do not change the physics. Do change everything else.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# D2Q9 lattice constants
#
#   6   2   5
#     \ | /
#   3 - 0 - 1
#     / | \
#   7   4   8
# ----------------------------------------------------------------------------
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)  # bounce-back partner
Q = 9

SIZES = {
    # name:      (nx,   ny,   nsteps)
    "tiny":      (100,   40,    500),
    "small":     (400,  100,   2000),
    "medium":    (1000, 250,  10000),
    "large":     (4000, 1000, 100000),
}


# ----------------------------------------------------------------------------
# Core kernels — naive on purpose
# ----------------------------------------------------------------------------
def equilibrium(rho, ux, uy):
    """Equilibrium distribution for every cell.

        f_eq_i = w_i * rho * (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u)

    Naive: a Python loop over the 9 directions, each allocating temporaries.
    """
    usqr = 1.5 * (ux * ux + uy * uy)
    feq = np.empty((Q,) + rho.shape, dtype=np.float64)
    for i in range(Q):
        cu = 3.0 * (C[i, 0] * ux + C[i, 1] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usqr)
    return feq


def macroscopic(f):
    """Density and velocity as moments of f. Fresh arrays on every call."""
    rho = np.zeros(f.shape[1:], dtype=np.float64)
    ux = np.zeros(f.shape[1:], dtype=np.float64)
    uy = np.zeros(f.shape[1:], dtype=np.float64)
    for i in range(Q):
        rho += f[i]
        ux += C[i, 0] * f[i]
        uy += C[i, 1] * f[i]
    ux /= rho
    uy /= rho
    return rho, ux, uy


def stream(f):
    """Shift every f_i one cell along direction i.

    Naive: np.roll per direction into a brand new array. This is the single
    biggest source of memory traffic in the whole solver.
    """
    out = np.empty_like(f)
    for i in range(Q):
        out[i] = np.roll(np.roll(f[i], C[i, 0], axis=0), C[i, 1], axis=1)
    return out


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------
def make_cylinder(nx, ny):
    """Obstacle mask for flow past a cylinder. Returns (solid, radius)."""
    cx, cy, r = nx // 5, ny // 2, max(3, ny // 9)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid = (x - cx) ** 2 + (y - cy) ** 2 < r * r
    solid[:, 0] = True      # no-slip walls
    solid[:, -1] = True
    return solid, r


def make_cavity(nx, ny):
    """Lid-driven cavity: solid on all four sides; the top row is the moving lid."""
    solid = np.zeros((nx, ny), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


def run(case="cylinder", nx=400, ny=100, nsteps=2000, reynolds=200.0,
        u0=0.05, sample_every=0, progress=True):
    """Run the solver. Returns (results, timing)."""
    if case == "cylinder":
        solid, r = make_cylinder(nx, ny)
        length_scale = 2 * r
    elif case == "cavity":
        solid = make_cavity(nx, ny)
        length_scale = ny - 2
    else:
        raise ValueError(f"unknown case {case!r}")

    nu = u0 * length_scale / reynolds          # kinematic viscosity
    tau = 3.0 * nu + 0.5                       # BGK relaxation time
    omega = 1.0 / tau
    if not (0.5 < tau < 2.5):
        raise ValueError(f"tau={tau:.3f} outside the usable range; adjust u0 or Re")

    # the lid row is driven, not bounced back
    lid = np.zeros_like(solid)
    if case == "cavity":
        lid[:, -1] = True
    bounce = solid & ~lid

    rho = np.ones((nx, ny), dtype=np.float64)
    ux = np.zeros((nx, ny), dtype=np.float64)
    uy = np.zeros((nx, ny), dtype=np.float64)
    if case == "cylinder":
        ux[:] = u0
        # tiny perturbation so the vortex street actually starts
        ux = ux * (1.0 + 1e-4 * np.sin(np.linspace(0, 2 * np.pi, ny))[None, :])
    ux[solid] = 0.0
    uy[solid] = 0.0
    f = equilibrium(rho, ux, uy)

    samples = []
    t_start = time.perf_counter()

    for step in range(nsteps):
        rho, ux, uy = macroscopic(f)

        # ---- boundary conditions
        if case == "cylinder":
            ux[0, :] = u0                      # inlet
            uy[0, :] = 0.0
            rho[0, :] = 1.0
        else:
            ux[:, -1] = u0                     # moving lid
            uy[:, -1] = 0.0
            rho[:, -1] = 1.0
        ux[bounce] = 0.0
        uy[bounce] = 0.0

        # ---- collision (BGK)
        feq = equilibrium(rho, ux, uy)
        if case == "cylinder":
            f[:, 0, :] = feq[:, 0, :]          # Dirichlet inlet
        else:
            f[:, :, -1] = feq[:, :, -1]        # Dirichlet lid
        fpost = f - omega * (f - feq)

        # ---- bounce-back inside solid cells (naive: loop + boolean indexing)
        for i in range(Q):
            fpost[i][bounce] = f[OPP[i]][bounce]

        # ---- streaming
        f = stream(fpost)

        # ---- zero-gradient outflow on the last column
        if case == "cylinder":
            for i in (3, 6, 7):
                f[i, -1, :] = f[i, -2, :]

        if sample_every and step % sample_every == 0:
            samples.append((step, float(uy[(2 * nx) // 5, ny // 2])))

        if progress and nsteps >= 20 and step % max(1, nsteps // 10) == 0:
            print(f"  step {step:>7d} / {nsteps}", flush=True)

    runtime = time.perf_counter() - t_start

    rho, ux, uy = macroscopic(f)
    ux[solid] = 0.0
    uy[solid] = 0.0

    results = {
        "ux": ux, "uy": uy, "rho": rho, "solid": solid,
        "probe": np.array(samples, dtype=np.float64) if samples else np.zeros((0, 2)),
    }
    meta = {"case": case, "nx": nx, "ny": ny, "nsteps": nsteps,
            "reynolds": reynolds, "u0": u0, "tau": tau}
    timing = {"runtime_s": runtime,
              "mlups": nx * ny * nsteps / runtime / 1e6,
              "cells": nx * ny, "steps": nsteps}
    return results, meta, timing


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def centreline_profiles(ux, uy):
    """Cavity validation: u along the vertical centreline, v along the horizontal."""
    nx, ny = ux.shape
    return ux[nx // 2, :], uy[:, ny // 2]


def strouhal(probe, length_scale, u0):
    """Strouhal number from the wake probe time series (cylinder case)."""
    if probe.shape[0] < 32:
        return float("nan")
    sig = probe[:, 1] - probe[:, 1].mean()
    dt = probe[1, 0] - probe[0, 0]
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, d=dt)
    spec[0] = 0.0
    return float(freqs[np.argmax(spec)] * length_scale / u0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["cylinder", "cavity"], default="cylinder")
    p.add_argument("--size", choices=list(SIZES), default="small")
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--nsteps", type=int)
    p.add_argument("--reynolds", type=float)
    p.add_argument("--u0", type=float, default=0.05)
    p.add_argument("--out", type=str)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    nx, ny, nsteps = SIZES[args.size]
    nx, ny = args.nx or nx, args.ny or ny
    nsteps = args.nsteps or nsteps
    re = args.reynolds if args.reynolds is not None else (
        200.0 if args.case == "cylinder" else 100.0)

    print(f"D2Q9 baseline | case={args.case} grid={nx}x{ny} steps={nsteps} Re={re}")
    results, meta, timing = run(
        case=args.case, nx=nx, ny=ny, nsteps=nsteps, reynolds=re, u0=args.u0,
        sample_every=10 if args.case == "cylinder" else 0, progress=not args.quiet)

    print(f"\nruntime : {timing['runtime_s']:.2f} s")
    print(f"MLUPS   : {timing['mlups']:.3f}")
    print(f"tau     : {meta['tau']:.4f}")
    print(f"max|ux| : {np.abs(results['ux']).max():.6f}")
    print(f"mean rho: {results['rho'][~results['solid']].mean():.8f}  (should stay ~1)")

    if args.case == "cylinder":
        _, r = make_cylinder(nx, ny)
        print(f"Strouhal: {strouhal(results['probe'], 2 * r, args.u0):.4f}"
              f"   (expect ~0.2 at Re=200 once shedding is developed)")
    else:
        u_vert, _ = centreline_profiles(results["ux"], results["uy"])
        print(f"u centreline min/max: {u_vert.min():+.6f} / {u_vert.max():+.6f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **results)
        out.with_suffix(".json").write_text(json.dumps({**timing, **meta}, indent=2))
        print(f"\nwrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
