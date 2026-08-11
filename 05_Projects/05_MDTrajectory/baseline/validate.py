#!/usr/bin/env python3
"""
Validate an MD analysis result against the baseline.

The RDF histogram is a count of integers: it must match exactly. Hydrogen-bond
counts are integers too. Q(t) is a ratio of integer counts, so it must match to
floating-point round-off. In other words, none of these have a legitimate
excuse to drift — if they do, the usual culprits are periodic-boundary handling
and histogram bin-edge conventions.

    python validate.py --size small --candidate ../results/analysis_gpu.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--baseline", default=None)
    p.add_argument("--candidate", default=None)
    args = p.parse_args()

    d = Path(args.data) / args.size
    base_path = Path(args.baseline) if args.baseline else d / "analysis_baseline.npz"
    base = np.load(base_path)

    print(f"SANITY  ({base_path})")
    ok = True
    if "rdf_g" in base.files:
        r, g = base["rdf_r"], base["rdf_g"]
        peak = int(np.argmax(g))
        print(f"  RDF first peak     : r = {r[peak]:.3f} nm, g = {g[peak]:.3f}")
        if not (0.24 <= r[peak] <= 0.34):
            print("    FAIL: first peak is not where a dense fluid puts it")
            ok = False
        # g(r) must go to zero at short range (excluded volume)
        if g[r < 0.15].max() > 0.05:
            print("    FAIL: non-zero g(r) inside the excluded volume — "
                  "check minimum-image handling")
            ok = False
    if "hbonds" in base.files:
        hb = base["hbonds"]
        rel = hb.std() / max(hb.mean(), 1e-9)
        print(f"  hydrogen bonds     : mean {hb.mean():.1f}, "
              f"frame-to-frame scatter {rel:.1%}")
        if rel > 0.25:
            print("    FAIL: hydrogen-bond count is not stable across frames")
            ok = False
    if "q" in base.files:
        q = base["q"]
        print(f"  Q(t)               : first {q[0]:.4f}, last {q[-1]:.4f}, "
              f"min {q.min():.4f}")
        if q.min() < 0.7:
            print("    FAIL: the protein came apart; contact map is meaningless")
            ok = False
    print(f"  {'PASS' if ok else 'FAIL'}")

    if not args.candidate:
        raise SystemExit(0 if ok else 1)

    cand = np.load(args.candidate)
    print(f"\nREGRESSION  (baseline vs {args.candidate})")
    checks = [
        ("rdf_hist", "exact", 0),
        ("hbonds", "exact", 0),
        ("q", "close", 1e-12),
        ("native_contacts", "exact", 0),
    ]
    for key, mode, tol in checks:
        if key not in base.files:
            continue
        if key not in cand.files:
            print(f"  {key:<16} MISSING from candidate")
            ok = False
            continue
        b, c = base[key], cand[key]
        if b.shape != c.shape:
            print(f"  {key:<16} FAIL shape {b.shape} vs {c.shape}")
            ok = False
            continue
        if mode == "exact":
            n_bad = int((b != c).sum())
            passed = n_bad == 0
            print(f"  {key:<16} {'PASS' if passed else 'FAIL'}  "
                  f"{n_bad:,} of {b.size:,} entries differ")
        else:
            err = float(np.abs(b - c).max())
            passed = err <= tol
            print(f"  {key:<16} {'PASS' if passed else 'FAIL'}  "
                  f"max abs diff {err:.3e} (tol {tol:.0e})")
        ok &= passed

    print("\n" + ("OK" if ok else "FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
