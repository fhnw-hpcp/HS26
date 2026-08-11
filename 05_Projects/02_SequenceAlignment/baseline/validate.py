#!/usr/bin/env python3
"""
Validate a Smith-Waterman result file.

1. REGRESSION -- scores must be BIT-IDENTICAL to the baseline. The scores are
   integers; there is no floating-point excuse. If you narrowed the datatype to
   int16/int8 and saturated, this is where you find out.

2. PLAUSIBILITY -- reported alignment end positions should sit near the true
   origin of each read, and scores should track the number of introduced
   mutations. This catches an implementation that is self-consistent but wrong.

    python validate.py --size small --candidate ../results/sw_gpu.tsv
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_scores(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["read"]] = (int(row["score"]), int(row["query_end"]),
                                int(row["target_end"]))
    return out


def load_truth(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["read"]] = (int(row["true_origin"]), int(row["n_subs"]),
                                int(row["n_indels"]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--baseline", default=None)
    p.add_argument("--candidate", default=None)
    args = p.parse_args()

    d = Path(args.data) / args.size
    base_path = Path(args.baseline) if args.baseline else d / "scores_baseline.tsv"
    base = load_scores(base_path)
    truth = load_truth(d / "truth.tsv")

    scores = np.array([v[0] for v in base.values()])
    print(f"PLAUSIBILITY  ({base_path})")
    print(f"  reads            : {len(base):,}")
    print(f"  score mean/min/max: {scores.mean():.2f} / {scores.min()} / {scores.max()}")

    # the alignment should end near the true origin plus the read length
    offsets, muts = [], []
    for name, (score, qe, te) in base.items():
        origin, n_sub, n_indel = truth[name]
        offsets.append(te - (origin + qe))
        muts.append(n_sub + n_indel)
    offsets = np.array(offsets)
    within = np.abs(offsets) <= 10
    print(f"  end position within 10 bp of truth: {within.mean():.1%}")
    corr = np.corrcoef(scores, np.array(muts))[0, 1]
    print(f"  corr(score, n_mutations)          : {corr:+.3f}  (should be clearly negative)")
    ok_plaus = within.mean() > 0.9 and corr < -0.3
    print(f"  {'PASS' if ok_plaus else 'FAIL'}")

    if not args.candidate:
        return
    cand = load_scores(args.candidate)
    print(f"\nREGRESSION  (baseline vs {args.candidate})")
    missing = set(base) - set(cand)
    if missing:
        print(f"  FAIL: candidate is missing {len(missing):,} reads")
        raise SystemExit(1)

    bad = [(n, base[n][0], cand[n][0]) for n in base if base[n][0] != cand[n][0]]
    if bad:
        print(f"  FAIL: {len(bad):,} of {len(base):,} scores differ")
        for n, b, c in bad[:10]:
            print(f"    {n}: baseline {b} vs candidate {c}")
        raise SystemExit(1)
    print(f"  PASS: all {len(base):,} scores bit-identical "
          f"(checksum {int(scores.sum())})")


if __name__ == "__main__":
    main()
