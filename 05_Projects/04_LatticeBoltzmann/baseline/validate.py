#!/usr/bin/env python3
"""
Validate a Lattice Boltzmann result against the baseline and against physics.

1. REGRESSION -- the velocity field must match a baseline run of the same case
   at the same resolution and step count. Single precision will not give you
   bit-identical results; report the tolerance you accept and defend it.

2. PHYSICS -- checks that survive any reimplementation:
     * mass conservation: mean density stays near 1
     * no-slip: velocity is exactly zero inside solid cells
     * cavity: the centreline profile is antisymmetric-ish and the primary
       vortex sits in the right place
     * cylinder: the Strouhal number of the shed wake is ~0.2 at Re=200

   The literature reference for the cavity is Ghia, Ghia & Shin (1982),
   J. Comput. Phys. 48, 387, Table I. Transcribing those values and comparing
   your centreline profile against them is part of the assignment — this script
   deliberately does not do it for you.

    python validate.py --baseline ref_cavity.npz --candidate cavity_gpu.npz
"""

import argparse

import numpy as np


def physics_checks(z, u0=0.05):
    ux, uy, rho, solid = z["ux"], z["uy"], z["rho"], z["solid"]
    ok = True

    mean_rho = float(rho[~solid].mean())
    drift = abs(mean_rho - 1.0)
    print(f"  mean density        : {mean_rho:.6f}  (drift {drift:.2%})")
    if drift > 0.05:
        print("    FAIL: mass drifted more than 5%")
        ok = False

    max_solid_u = float(max(np.abs(ux[solid]).max(initial=0.0),
                            np.abs(uy[solid]).max(initial=0.0)))
    print(f"  max |u| in solid    : {max_solid_u:.3e}")
    if max_solid_u > 1e-12:
        print("    FAIL: no-slip violated inside obstacle cells")
        ok = False

    umax = float(np.abs(ux).max())
    print(f"  max |ux|            : {umax:.6f}")
    if not np.isfinite(umax) or umax > 0.5:
        print("    FAIL: velocity is not finite / far above the lattice Mach limit")
        ok = False

    nx, ny = ux.shape
    u_vert = ux[nx // 2, :]
    print(f"  centreline u min/max: {u_vert.min():+.6f} / {u_vert.max():+.6f}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", default=None)
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance on the velocity field")
    args = p.parse_args()

    base = np.load(args.baseline)
    print(f"PHYSICS  ({args.baseline})")
    ok = physics_checks(base)

    if args.candidate:
        cand = np.load(args.candidate)
        print(f"\nPHYSICS  ({args.candidate})")
        ok &= physics_checks(cand)

        print(f"\nREGRESSION  (rtol {args.rtol:.0e})")
        for key in ("ux", "uy", "rho"):
            b, c = base[key], cand[key]
            if b.shape != c.shape:
                print(f"  {key}: FAIL shape {b.shape} vs {c.shape}")
                ok = False
                continue
            scale = max(float(np.abs(b).max()), 1e-12)
            err = float(np.abs(b - c).max() / scale)
            rms = float(np.sqrt(((b - c) ** 2).mean()) / scale)
            passed = err <= args.rtol
            ok &= passed
            print(f"  {key}: {'PASS' if passed else 'FAIL'} "
                  f"max rel err {err:.3e}, rms {rms:.3e}")

    print("\n" + ("OK" if ok else "FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
