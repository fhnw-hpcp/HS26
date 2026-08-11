#!/usr/bin/env python3
"""
Baseline Smith-Waterman local alignment — deliberately naive, correct, and slow.

This is the code you are asked to make fast. The scoring matrix is filled cell
by cell in a Python loop, exactly as the recurrence is written in the textbook:

    H[i,j] = max( 0,
                  H[i-1,j-1] + s(a_i, b_j),   # match / mismatch
                  E[i,j],                     # gap in the query  (affine)
                  F[i,j] )                    # gap in the target (affine)

Affine gaps are handled with the Gotoh three-matrix formulation, because that
is what real aligners use and it is what makes the dependency structure
interesting.

Usage
-----
    python smith_waterman.py --size small
    python smith_waterman.py --reads ../data/reads.fasta --ref ../data/ref.fasta

Performance is reported in MCUPS (mega cell updates per second):
    MCUPS = sum(len(read) * len(window)) / runtime / 1e6

The workload models the *realignment* stage of a read mapper: each read has
already been seeded to a candidate window of the reference, and the exact
Smith-Waterman score for that read/window pair has to be computed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# scoring scheme (integers, as in every production aligner)
MATCH = 2
MISMATCH = -1
GAP_OPEN = 3      # penalty applied when a gap is opened
GAP_EXTEND = 1    # penalty per extended position

SIZES = {
    # name:     (n_reads, read_len, window_len)
    "tiny":     (20,   100,   400),
    "small":    (200,  150,  1000),
    "medium":   (2000, 150,  2000),
    "large":    (20000, 250, 4000),
}


# ----------------------------------------------------------------------------
# FASTA I/O
# ----------------------------------------------------------------------------
def read_fasta(path):
    """Yield (name, sequence) pairs."""
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:], []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


# ----------------------------------------------------------------------------
# The kernel — naive on purpose
# ----------------------------------------------------------------------------
def smith_waterman(query, target,
                   match=MATCH, mismatch=MISMATCH,
                   gap_open=GAP_OPEN, gap_extend=GAP_EXTEND):
    """Optimal local alignment score with affine gaps (Gotoh).

    Returns (score, end_i, end_j) where the ends are the 1-based coordinates of
    the highest-scoring cell.

    Naive: one Python-level iteration per matrix cell. Three rows are kept, so
    memory is O(n) — the traceback is deliberately not computed here.
    """
    m, n = len(query), len(target)
    # previous / current rows of H (best), E (gap in query), F (gap in target)
    h_prev = [0] * (n + 1)
    h_cur = [0] * (n + 1)
    e_prev = [0] * (n + 1)
    e_cur = [0] * (n + 1)

    best, best_i, best_j = 0, 0, 0

    for i in range(1, m + 1):
        qi = query[i - 1]
        f = 0                       # gap in target, runs along the row
        h_cur[0] = 0
        e_cur[0] = 0
        for j in range(1, n + 1):
            # diagonal: match or mismatch
            sub = match if qi == target[j - 1] else mismatch
            diag = h_prev[j - 1] + sub

            # gap in the query (vertical move): open from H or extend E
            e = e_prev[j] - gap_extend
            o = h_prev[j] - gap_open - gap_extend
            if o > e:
                e = o
            if e < 0:
                e = 0
            e_cur[j] = e

            # gap in the target (horizontal move): open from H or extend F
            o = h_cur[j - 1] - gap_open - gap_extend
            f = f - gap_extend
            if o > f:
                f = o
            if f < 0:
                f = 0

            # local alignment: never go below zero
            h = diag
            if e > h:
                h = e
            if f > h:
                h = f
            if h < 0:
                h = 0
            h_cur[j] = h

            if h > best:
                best, best_i, best_j = h, i, j

        h_prev, h_cur = h_cur, h_prev
        e_prev, e_cur = e_cur, e_prev

    return best, best_i, best_j


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def align_all(reads, reference, windows, progress=True):
    """Align every read against its candidate window. Returns results + timing."""
    results = []
    cells = 0
    t0 = time.perf_counter()
    for k, (name, seq) in enumerate(reads):
        start, stop = windows[name]
        window = reference[start:stop]
        score, ei, ej = smith_waterman(seq, window)
        results.append((name, score, ei, start + ej))
        cells += len(seq) * len(window)
        if progress and len(reads) >= 10 and k % max(1, len(reads) // 10) == 0:
            print(f"  read {k:>6d} / {len(reads)}", flush=True)
    runtime = time.perf_counter() - t0
    timing = {"runtime_s": runtime,
              "cells": cells,
              "mcups": cells / runtime / 1e6,
              "n_reads": len(reads)}
    return results, timing


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data", help="directory with generated data")
    p.add_argument("--size", choices=list(SIZES), default="small",
                   help="which generated dataset to use")
    p.add_argument("--limit", type=int, help="only align the first N reads")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    data = Path(args.data) / args.size
    if not data.exists():
        raise SystemExit(
            f"{data} not found — run:  python generate_data.py --size {args.size}")

    ref = next(iter(read_fasta(data / "reference.fasta")))[1]
    reads = list(read_fasta(data / "reads.fasta"))
    windows = {k: tuple(v) for k, v in
               json.loads((data / "windows.json").read_text()).items()}
    if args.limit:
        reads = reads[:args.limit]

    print(f"Smith-Waterman baseline | {len(reads)} reads, "
          f"reference {len(ref):,} bp, window {windows[reads[0][0]][1] - windows[reads[0][0]][0]:,} bp")

    results, timing = align_all(reads, ref, windows, progress=not args.quiet)

    scores = np.array([r[1] for r in results])
    print(f"\nruntime : {timing['runtime_s']:.2f} s")
    print(f"MCUPS   : {timing['mcups']:.3f}")
    print(f"cells   : {timing['cells']:,}")
    print(f"score   : mean {scores.mean():.2f}  min {scores.min()}  max {scores.max()}")
    print(f"checksum: {int(scores.sum())}   <-- must be identical after optimization")

    out = Path(args.out) if args.out else data / "scores_baseline.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("read\tscore\tquery_end\ttarget_end\n")
        for name, score, ei, ej in results:
            fh.write(f"{name}\t{score}\t{ei}\t{ej}\n")
    out.with_suffix(".json").write_text(json.dumps(
        {**timing, "checksum": int(scores.sum())}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
