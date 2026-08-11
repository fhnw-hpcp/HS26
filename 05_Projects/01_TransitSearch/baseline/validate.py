#!/usr/bin/env python3
"""
Validate a BLS result file.

Two independent checks:

1. RECOVERY  -- did the search find the injected planets? A detection counts if
   SDE exceeds the threshold and the recovered period matches the true period
   within tolerance, allowing for the usual 2x / 0.5x period aliases.

2. REGRESSION -- does an optimized result file agree with the baseline result
   file? This is the check that matters for the assignment: your fast version
   must agree with the slow one.

    python validate.py --size small
    python validate.py --size small --candidate ../results/bls_gpu.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

# Detection threshold. On the generated data, injected planets land at SDE 25-60
# while planet-free stars peak below 8, so 10 separates them cleanly. Real
# surveys use a similar value and then vet the candidates by hand.
SDE_THRESHOLD = 10.0
PERIOD_RTOL = 0.02      # 2 per cent


def load(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["star_id"])] = {k: float(v) for k, v in row.items()
                                        if k != "star_id"}
    return out


def load_truth(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["star_id"])] = {
                "has_planet": int(row["has_planet"]),
                "period": float(row["period"]) if row["period"] != "nan" else np.nan,
                "depth": float(row["depth"]) if row["depth"] != "nan" else np.nan,
            }
    return out


def period_matches(found, true):
    """Allow the fundamental and the usual 2x / 0.5x aliases."""
    for factor in (1.0, 2.0, 0.5, 3.0, 1 / 3):
        if abs(found - true * factor) <= PERIOD_RTOL * true * factor:
            return True
    return False


def recovery(results, truth):
    tp = fp = fn = tn = 0
    recovered = []
    for sid, tr in truth.items():
        if sid not in results:
            continue
        res = results[sid]
        detected = res["sde"] >= SDE_THRESHOLD
        if tr["has_planet"]:
            if detected and period_matches(res["period"], tr["period"]):
                tp += 1
                recovered.append((sid, tr["period"], res["period"], res["sde"]))
            else:
                fn += 1
        else:
            fp += 1 if detected else 0
            tn += 1 if not detected else 0
    return tp, fp, fn, tn, recovered


def regression(baseline, candidate):
    """Compare two result files star by star."""
    missing = set(baseline) - set(candidate)
    if missing:
        print(f"  FAIL: candidate is missing {len(missing)} stars")
        return False
    ok = True
    worst_p = worst_sde = 0.0
    mismatched = []
    for sid, base in baseline.items():
        cand = candidate[sid]
        dp = abs(cand["period"] - base["period"]) / base["period"]
        ds = abs(cand["sde"] - base["sde"]) / max(base["sde"], 1e-9)
        worst_p = max(worst_p, dp)
        worst_sde = max(worst_sde, ds)
        if dp > PERIOD_RTOL:
            mismatched.append((sid, base["period"], cand["period"]))
            ok = False
    print(f"  worst relative period difference: {worst_p:.3e}")
    print(f"  worst relative SDE difference   : {worst_sde:.3e}")
    if mismatched:
        print(f"  FAIL: {len(mismatched)} stars disagree beyond {PERIOD_RTOL:.0%}")
        for sid, b, c in mismatched[:10]:
            print(f"    star {sid}: baseline {b:.6f} vs candidate {c:.6f}")
    else:
        print(f"  PASS: all {len(baseline)} stars agree within {PERIOD_RTOL:.0%}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--baseline", default=None)
    p.add_argument("--candidate", default=None,
                   help="optimized result file to compare against the baseline")
    args = p.parse_args()

    d = Path(args.data) / args.size
    base_path = Path(args.baseline) if args.baseline else d / "bls_baseline.csv"
    baseline = load(base_path)
    truth = load_truth(d / "truth.csv")

    print(f"RECOVERY  ({base_path})")
    tp, fp, fn, tn, rec = recovery(baseline, truth)
    n_planets = tp + fn
    print(f"  injected planets : {n_planets}")
    print(f"  recovered        : {tp}  ({tp / max(n_planets, 1):.0%})")
    print(f"  missed           : {fn}")
    print(f"  false positives  : {fp} of {fp + tn} planet-free stars")
    if rec:
        print("  examples (star, true P, found P, SDE):")
        for sid, tp_, fp_, s_ in rec[:5]:
            print(f"    {sid:>4d}  {tp_:8.4f}  {fp_:8.4f}  {s_:6.2f}")

    if args.candidate:
        print(f"\nREGRESSION  (baseline vs {args.candidate})")
        ok = regression(baseline, load(args.candidate))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
