#!/usr/bin/env python3
"""
Validate a climate index result file against the baseline.

Percentile-based indices are the classic place where two implementations
silently disagree, because np.percentile, np.quantile and the ETCCDI
bootstrapped definition do not all interpolate the same way. This script
reports the worst absolute and relative difference per index so you can see
whether you have a real bug or an interpolation convention mismatch.

    python validate.py --size small --candidate ../results/indices_dask.npz
"""

import argparse
from pathlib import Path

import numpy as np

# per-index tolerance: tx90p is a fraction, hwdi/cdd are integer day counts,
# gdd is a large sum so it gets a relative tolerance
TOL = {
    "tx90p": ("abs", 1e-9),
    "hwdi":  ("abs", 1e-9),
    "cdd":   ("abs", 1e-9),
    "gdd":   ("rel", 1e-10),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--baseline", default=None)
    p.add_argument("--candidate", required=True)
    args = p.parse_args()

    d = Path(args.data) / args.size
    base_path = Path(args.baseline) if args.baseline else d / "indices_baseline.npz"
    base = np.load(base_path)
    cand = np.load(args.candidate)

    print(f"baseline : {base_path}")
    print(f"candidate: {args.candidate}\n")

    ok = True
    for key in base.files:
        if key not in cand.files:
            print(f"{key:<8} MISSING from candidate")
            ok = False
            continue
        b, c = base[key], cand[key]
        if b.shape != c.shape:
            print(f"{key:<8} FAIL shape {b.shape} vs {c.shape}")
            ok = False
            continue

        valid = np.isfinite(b) & np.isfinite(c)
        if valid.sum() == 0:
            print(f"{key:<8} FAIL no finite values in common")
            ok = False
            continue
        if (np.isfinite(b) != np.isfinite(c)).any():
            n = int((np.isfinite(b) != np.isfinite(c)).sum())
            print(f"{key:<8} WARN {n} cells differ in NaN pattern")

        diff = np.abs(b[valid] - c[valid])
        mode, tol = TOL.get(key, ("rel", 1e-9))
        if mode == "rel":
            scale = np.maximum(np.abs(b[valid]), 1e-12)
            metric = (diff / scale).max()
            label = "max rel diff"
        else:
            metric = diff.max()
            label = "max abs diff"

        passed = metric <= tol
        ok &= passed
        n_bad = int((diff > 0).sum())
        print(f"{key:<8} {'PASS' if passed else 'FAIL'}  {label} {metric:.3e} "
              f"(tol {tol:.0e}), {n_bad:,} of {valid.sum():,} cells differ at all")

    print("\n" + ("ALL INDICES MATCH" if ok else
                  "MISMATCH — check your percentile interpolation and "
                  "run-length boundary handling first"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
