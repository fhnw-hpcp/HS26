#!/usr/bin/env python3
"""
Generate the sequence-alignment dataset: a reference genome and a set of reads
sampled from it with realistic substitutions and indels.

Everything is generated from a fixed seed, so the dataset is reproducible and
nobody has to download anything to get started.

    python generate_data.py --size small
    python generate_data.py --size medium --out ../data

To work with the real thing instead, drop a genome FASTA in place of
reference.fasta and regenerate the reads with --from-fasta. The E. coli K-12
MG1655 reference (~4.6 Mbp) is at
https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3 and real Illumina reads are at
https://www.ncbi.nlm.nih.gov/sra — but do that *after* your pipeline works.

Outputs, per size, into <out>/<size>/ :
    reference.fasta   the reference sequence
    reads.fasta       the reads
    windows.json      read name -> [start, stop] candidate window in the reference
    truth.tsv         read name -> true origin, n_subs, n_indels
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASES = np.array(list("ACGT"))

SIZES = {
    # name:     (n_reads, read_len, window_len, ref_len)
    "tiny":     (20,    100,   400,    50_000),
    "small":    (200,   150,  1000,   200_000),
    "medium":   (2000,  150,  2000, 1_000_000),
    "large":    (20000, 250,  4000, 5_000_000),
}

SUB_RATE = 0.02      # per-base substitution probability
INDEL_RATE = 0.004   # per-base indel probability
MAX_INDEL = 4


def make_reference(n, rng):
    """A random reference with a bit of local GC structure, so it is not uniform."""
    gc = 0.35 + 0.25 * (np.sin(np.linspace(0, 40 * np.pi, n)) * 0.5 + 0.5)
    u = rng.random(n)
    seq = np.empty(n, dtype="<U1")
    # A/T where u is low, G/C where u is high, with the local GC fraction as split
    is_gc = u < gc
    seq[~is_gc] = np.where(rng.random((~is_gc).sum()) < 0.5, "A", "T")
    seq[is_gc] = np.where(rng.random(is_gc.sum()) < 0.5, "G", "C")
    return "".join(seq)


def mutate(sub, rng):
    """Apply substitutions and indels to a subsequence. Returns (read, n_sub, n_indel)."""
    out = []
    n_sub = n_indel = 0
    i = 0
    while i < len(sub):
        r = rng.random()
        if r < INDEL_RATE:
            n_indel += 1
            length = int(rng.integers(1, MAX_INDEL + 1))
            if rng.random() < 0.5:                     # deletion
                i += length
            else:                                      # insertion
                out.extend(rng.choice(BASES, size=length))
                i += 1
                if i <= len(sub):
                    out.append(sub[i - 1])
        elif r < INDEL_RATE + SUB_RATE:
            n_sub += 1
            alt = [b for b in "ACGT" if b != sub[i]]
            out.append(alt[int(rng.integers(0, 3))])
            i += 1
        else:
            out.append(sub[i])
            i += 1
    return "".join(out), n_sub, n_indel


def write_fasta(path, records, width=80):
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for k in range(0, len(seq), width):
                fh.write(seq[k:k + width] + "\n")


def generate(size, out_dir, seed=20260810):
    n_reads, read_len, window_len, ref_len = SIZES[size]
    rng = np.random.default_rng(seed)

    print(f"[{size}] reference: {ref_len:,} bp")
    ref = make_reference(ref_len, rng)

    reads, windows, truth = [], {}, []
    print(f"[{size}] reads: {n_reads:,} x {read_len} bp, window {window_len:,} bp")
    for k in range(n_reads):
        origin = int(rng.integers(0, ref_len - read_len - 1))
        raw = ref[origin:origin + read_len]
        read, n_sub, n_indel = mutate(raw, rng)
        name = f"read{k:06d}"
        reads.append((name, read))

        # candidate window: contains the true origin, but not centred on it,
        # which is what a seeding stage would realistically hand you
        offset = int(rng.integers(0, max(1, window_len - read_len)))
        start = max(0, min(origin - offset, ref_len - window_len))
        windows[name] = [start, start + window_len]
        truth.append((name, origin, n_sub, n_indel))

    d = Path(out_dir) / size
    d.mkdir(parents=True, exist_ok=True)
    write_fasta(d / "reference.fasta", [(f"ref_{size}", ref)])
    write_fasta(d / "reads.fasta", reads)
    (d / "windows.json").write_text(json.dumps(windows))
    with open(d / "truth.tsv", "w") as fh:
        fh.write("read\ttrue_origin\tn_subs\tn_indels\n")
        for name, origin, ns, ni in truth:
            fh.write(f"{name}\t{origin}\t{ns}\t{ni}\n")

    total_cells = n_reads * read_len * window_len
    print(f"[{size}] wrote {d}")
    print(f"[{size}] total DP cells: {total_cells:,}")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=list(SIZES) + ["all"], default="small")
    p.add_argument("--out", default="../data")
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()

    sizes = list(SIZES) if args.size == "all" else [args.size]
    for s in sizes:
        generate(s, args.out, args.seed)


if __name__ == "__main__":
    main()
