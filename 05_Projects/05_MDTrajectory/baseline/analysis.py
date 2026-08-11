#!/usr/bin/env python3
"""
Baseline MD trajectory analysis — deliberately naive, correct, and slow.

This is the code you are asked to make fast. Three analyses are run over the
trajectory, frame by frame, in the style of every first-draft analysis script:

  rdf       radial distribution function g(r) of water oxygens
            -- all pairs, O(N^2), one Python iteration per atom
  hbonds    hydrogen bonds by the standard geometric criterion
            (donor-acceptor distance < 0.35 nm and D-H...A angle > 150 deg)
  contacts  protein contact map, and the fraction of native contacts Q(t)

The trajectory is streamed one frame at a time, which is what MDAnalysis does
and what keeps memory bounded. Whether that is the right choice once you
parallelize is one of the questions you are asked to answer.

Usage
-----
    python analysis.py --size small
    python analysis.py --size small --analyses rdf --frames 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

RDF_RMAX = 1.2          # nm
RDF_NBINS = 240
HB_DIST = 0.35          # nm, donor-acceptor cutoff
HB_ANGLE = 150.0        # degrees, minimum D-H...A angle
CONTACT_CUT = 0.8       # nm, protein contact cutoff
NATIVE_CUT = 0.8        # nm, native contact definition from frame 0


def minimum_image(d, box):
    """Wrap displacement vectors into [-box/2, box/2)."""
    return d - box * np.round(d / box)


# ----------------------------------------------------------------------------
# Analyses — naive on purpose
# ----------------------------------------------------------------------------
def rdf_frame(pos, box, rmax=RDF_RMAX, nbins=RDF_NBINS):
    """Histogram of all pairwise distances.

    Naive: a Python loop over atoms, each iteration computing the distance to
    every remaining atom. O(N^2) work in O(N) Python iterations.
    """
    n = pos.shape[0]
    hist = np.zeros(nbins, dtype=np.int64)
    edges = np.linspace(0.0, rmax, nbins + 1)
    for i in range(n - 1):
        d = minimum_image(pos[i + 1:] - pos[i], box)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        hist += np.histogram(r, bins=edges)[0]
    return hist, edges


def normalize_rdf(hist, edges, n_atoms, n_frames, box):
    """Turn the pair histogram into g(r)."""
    volume = float(np.prod(box))
    r_lo, r_hi = edges[:-1], edges[1:]
    shell = 4.0 / 3.0 * np.pi * (r_hi ** 3 - r_lo ** 3)
    n_pairs = 0.5 * n_atoms * (n_atoms - 1)
    ideal = shell * n_pairs / volume
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / (ideal * n_frames)
    return 0.5 * (r_lo + r_hi), np.nan_to_num(g)


def hbonds_frame(pos, donors, hydrogens, acceptors, box,
                 dist_cut=HB_DIST, angle_cut=HB_ANGLE):
    """Count hydrogen bonds by the geometric criterion.

    Naive: a Python loop over donors, vectorized over acceptors, then an
    explicit loop over the surviving candidate pairs to test the angle.
    """
    count = 0
    cos_cut = np.cos(np.radians(angle_cut))
    # nearest hydrogen for each donor (waters have two; take whichever is closer)
    for di, d_idx in enumerate(donors):
        dv = minimum_image(pos[acceptors] - pos[d_idx], box)
        r = np.sqrt(np.einsum("ij,ij->i", dv, dv))
        cand = np.where((r < dist_cut) & (r > 1e-6))[0]
        if cand.size == 0:
            continue
        h_here = hydrogens[(hydrogens > d_idx) & (hydrogens <= d_idx + 2)]
        if h_here.size == 0:
            continue
        for c in cand:
            a_idx = acceptors[c]
            if a_idx == d_idx:
                continue
            for h_idx in h_here:
                dh = minimum_image(pos[h_idx] - pos[d_idx], box)
                ha = minimum_image(pos[a_idx] - pos[h_idx], box)
                ndh = np.linalg.norm(dh)
                nha = np.linalg.norm(ha)
                if ndh < 1e-6 or nha < 1e-6:
                    continue
                # angle at H between H->D and H->A
                cos_theta = float(np.dot(-dh, ha) / (ndh * nha))
                if cos_theta <= cos_cut:
                    count += 1
                    break
    return count


def contacts_frame(pos, box, cutoff=CONTACT_CUT):
    """Boolean contact map of the protein atoms. Naive O(N^2) loop."""
    n = pos.shape[0]
    cmap = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        d = minimum_image(pos[i + 1:] - pos[i], box)
        r2 = np.einsum("ij,ij->i", d, d)
        close = r2 < cutoff ** 2
        cmap[i, i + 1:] = close
        cmap[i + 1:, i] = close
    return cmap


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(data_dir, analyses, max_frames=None, progress=True):
    top = np.load(data_dir / "topology.npz")
    traj = np.load(data_dir / "trajectory.npy", mmap_mode="r")
    box = top["box"].astype(np.float64)
    types = top["types"]
    donors, acceptors, hydrogens = top["donors"], top["acceptors"], top["hydrogens"]

    water_o = np.where(types == 1)[0]
    protein = np.where(types == 0)[0]

    n_frames = traj.shape[0] if max_frames is None else min(max_frames, traj.shape[0])
    n_atoms = traj.shape[1]

    print(f"trajectory: {traj.shape[0]} frames x {n_atoms:,} atoms "
          f"({traj.nbytes / 1e6:.1f} MB), analysing {n_frames} frames")
    print(f"  water oxygens: {water_o.size:,}   protein: {protein.size:,}   "
          f"donors: {donors.size:,}   acceptors: {acceptors.size:,}")

    rdf_hist = np.zeros(RDF_NBINS, dtype=np.int64)
    rdf_edges = None
    hb_counts = []
    q_values = []
    native = None

    t_io = t_rdf = t_hb = t_ct = 0.0
    t0 = time.perf_counter()

    for frame in range(n_frames):
        t_a = time.perf_counter()
        pos = np.asarray(traj[frame], dtype=np.float64)   # <- the streaming read
        t_io += time.perf_counter() - t_a

        if "rdf" in analyses:
            t_a = time.perf_counter()
            h, rdf_edges = rdf_frame(pos[water_o], box)
            rdf_hist += h
            t_rdf += time.perf_counter() - t_a

        if "hbonds" in analyses:
            t_a = time.perf_counter()
            hb_counts.append(hbonds_frame(pos, donors, hydrogens, acceptors, box))
            t_hb += time.perf_counter() - t_a

        if "contacts" in analyses:
            t_a = time.perf_counter()
            cmap = contacts_frame(pos[protein], box)
            if native is None:
                native = cmap.copy()
                q_values.append(1.0)
            else:
                q_values.append(float((cmap & native).sum() / max(native.sum(), 1)))
            t_ct += time.perf_counter() - t_a

        if progress and n_frames >= 10 and frame % max(1, n_frames // 10) == 0:
            print(f"  frame {frame:>6d} / {n_frames}", flush=True)

    runtime = time.perf_counter() - t0

    results = {}
    if "rdf" in analyses:
        r, g = normalize_rdf(rdf_hist, rdf_edges, water_o.size, n_frames, box)
        results["rdf_r"] = r
        results["rdf_g"] = g
        results["rdf_hist"] = rdf_hist
    if "hbonds" in analyses:
        results["hbonds"] = np.array(hb_counts, dtype=np.int64)
    if "contacts" in analyses:
        results["q"] = np.array(q_values)
        results["native_contacts"] = np.array([int(native.sum() // 2)])

    timing = {"runtime_s": runtime, "io_s": t_io, "rdf_s": t_rdf,
              "hbonds_s": t_hb, "contacts_s": t_ct,
              "n_frames": n_frames, "n_atoms": int(n_atoms),
              "s_per_frame": runtime / n_frames}
    return results, timing


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--analyses", default="rdf,hbonds,contacts",
                   help="comma-separated subset of rdf,hbonds,contacts")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    d = Path(args.data) / args.size
    if not d.exists():
        raise SystemExit(
            f"{d} not found — run:  python generate_data.py --size {args.size}")

    analyses = [a.strip() for a in args.analyses.split(",") if a.strip()]
    results, timing = run(d, analyses, args.frames, progress=not args.quiet)

    print(f"\nruntime   : {timing['runtime_s']:.2f} s "
          f"({timing['s_per_frame']:.3f} s/frame)")
    print(f"  I/O     : {timing['io_s']:.2f} s  "
          f"({timing['io_s'] / timing['runtime_s']:.0%})")
    for key, label in (("rdf_s", "RDF"), ("hbonds_s", "hbonds"),
                       ("contacts_s", "contacts")):
        if timing[key] > 0:
            print(f"  {label:<8}: {timing[key]:.2f} s  "
                  f"({timing[key] / timing['runtime_s']:.0%})")

    if "rdf_g" in results:
        g, r = results["rdf_g"], results["rdf_r"]
        peak = int(np.argmax(g))
        print(f"\nRDF first peak at r = {r[peak]:.3f} nm, g = {g[peak]:.3f}   "
              f"(expect ~0.27-0.29 nm; the generated fluid is soft-sphere, so "
              f"g_max ~1.4 rather than real water's ~2.8)")
        print(f"RDF histogram checksum: {int(results['rdf_hist'].sum())}")
    if "hbonds" in results:
        hb = results["hbonds"]
        print(f"hydrogen bonds: mean {hb.mean():.1f} per frame "
              f"(min {hb.min()}, max {hb.max()})")
    if "q" in results:
        q = results["q"]
        print(f"native contacts: {int(results['native_contacts'][0])}, "
              f"Q(final) = {q[-1]:.4f}")

    out = Path(args.out) if args.out else d / "analysis_baseline.npz"
    np.savez_compressed(out, **results)
    Path(str(out).replace(".npz", ".json")).write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
