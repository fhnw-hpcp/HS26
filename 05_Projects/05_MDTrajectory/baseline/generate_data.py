#!/usr/bin/env python3
"""
Generate a molecular dynamics trajectory to analyse.

This is a *synthetic* trajectory, not a physically meaningful simulation: a
compact bonded "protein" surrounded by three-site "water" molecules in a
periodic box, evolved by overdamped relaxation of a soft-sphere potential with
thermal noise. It is built that way on purpose — overdamped relaxation is
unconditionally stable, so the generator always produces a usable trajectory,
and the structural features the analyses look for are correct by construction:

  * water oxygens have a first-neighbour shell at ~0.28 nm, so g(r) has its
    first peak in the right place. It is a soft-sphere fluid, so the peak is
    broader and lower (g ~ 1.4) than real water (g ~ 2.8) — the position is
    physical, the amplitude is not.
  * one hydrogen of each water is aimed at its nearest oxygen, so the D-H...A
    angle criterion actually fires and the hydrogen-bond count is stable frame
    to frame (~1.1 per water; real water donates ~2)
  * the protein is restrained to its initial fold, so Q(t) stays above ~0.95
    and a contact map means something

What is being benchmarked is the *analysis*, not the dynamics.

    python generate_data.py --size small
    python generate_data.py --size medium --out ../data

Outputs, per size, into <out>/<size>/ :
    trajectory.npy   float32 memmap, shape (n_frames, n_atoms, 3), nanometres
    topology.npz     atom names, types, residue ids, donor/acceptor indices, box

The trajectory is a flat memmap on purpose: it is the simplest thing that
streams, and swapping in a real compressed XTC via MDAnalysis is one of the
optimizations you are invited to evaluate. Real trajectories are at
https://www.mdanalysis.org/MDAnalysisData/ and https://markovmodel.github.io/mdshare/
"""

import argparse
from pathlib import Path

import numpy as np

SIZES = {
    # name:    (n_protein, n_waters, n_frames)
    "tiny":    (60,      400,   20),
    "small":   (300,   2_000,  100),
    "medium":  (900,   8_000,  400),
    "large":   (3000, 40_000, 2000),
}

WATER_DENSITY = 33.4   # molecules per nm^3, i.e. real liquid water
R_MIN = 0.288          # nm, soft-sphere exclusion radius; sets the RDF first peak
K_REP = 60.0           # soft repulsion stiffness (ETA*K_REP must stay below ~0.5)
K_BOND = 60.0          # harmonic bond stiffness
K_REST = 8.0           # protein restraint to the reference fold
ETA = 0.004            # overdamped step size
NOISE = 0.012          # nm, thermal displacement per frame
SWEEPS = 4             # relaxation sweeps per saved frame
BOND_R0 = 0.153        # nm, C-C
OH = 0.098             # nm, O-H


def minimum_image(d, box):
    """Wrap displacement vectors into [-box/2, box/2)."""
    return d - box * np.round(d / box)


# ----------------------------------------------------------------------------
# System construction
# ----------------------------------------------------------------------------
def build_system(n_protein, n_waters, box, rng):
    pos = []
    names, types, resids = [], [], []
    donors, acceptors, hydrogens = [], [], []
    bonds = []

    # --- "protein": a chain collapsed into a globule at the box centre
    centre = np.array([box / 2] * 3)
    p = centre + rng.normal(0, 0.15, 3)
    for i in range(n_protein):
        step = rng.normal(0, 1, 3)
        step /= np.linalg.norm(step)
        p = p + step * BOND_R0
        p = p + 0.08 * (centre - p)          # keep it compact
        pos.append(p.copy())
        names.append("CA" if i % 3 == 0 else "C")
        types.append(0)
        resids.append(i // 3)
        if i > 0:
            bonds.append((i - 1, i))
        # the protein has no explicit polar hydrogens in this model, so its
        # atoms act as acceptors only
        if i % 3 == 1:
            acceptors.append(i)

    # --- water oxygens on a jittered lattice, protein clashes removed
    protein_pos = np.array(pos)
    side = int(np.ceil((n_waters * 2.2) ** (1 / 3)))
    grid = np.linspace(0, box, side, endpoint=False)
    sites = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    sites = sites + rng.normal(0, box / side * 0.10, sites.shape)
    rng.shuffle(sites)
    d = minimum_image(sites[:, None, :] - protein_pos[None, :, :], box)
    clash = (np.einsum("ijk,ijk->ij", d, d) < 0.30 ** 2).any(axis=1)
    sites = sites[~clash][:n_waters]
    if sites.shape[0] < n_waters:
        raise RuntimeError(f"only placed {sites.shape[0]} of {n_waters} waters; "
                           f"increase the lattice oversampling factor")

    for w in range(n_waters):
        o = sites[w] % box
        idx_o = len(pos)
        pos.append(o)
        names.append("OW")
        types.append(1)
        resids.append(10_000 + w)
        acceptors.append(idx_o)
        donors.append(idx_o)
        for h in range(2):
            v = rng.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            pos.append((o + v * OH) % box)
            names.append(f"HW{h + 1}")
            types.append(2)
            resids.append(10_000 + w)
            hydrogens.append(len(pos) - 1)
            bonds.append((idx_o, len(pos) - 1))

    return (np.array(pos, dtype=np.float64),
            np.array(names), np.array(types), np.array(resids),
            np.array(bonds, dtype=np.int64),
            np.array(donors), np.array(acceptors), np.array(hydrogens))


# ----------------------------------------------------------------------------
# Overdamped relaxation
# ----------------------------------------------------------------------------
def cell_pairs(pos, box, cutoff):
    """Neighbour pair list (i, j) with i < j, built with a cell list."""
    n = pos.shape[0]
    ncell = max(3, int(box / cutoff))
    cs = box / ncell
    cell = np.floor(pos / cs).astype(np.int64) % ncell
    flat = (cell[:, 0] * ncell + cell[:, 1]) * ncell + cell[:, 2]
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(ncell ** 3 + 1))

    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    ii, jj = [], []
    for cid in range(ncell ** 3):
        a0, a1 = starts[cid], starts[cid + 1]
        if a1 <= a0:
            continue
        me = order[a0:a1]
        cz = cid % ncell
        cy = (cid // ncell) % ncell
        cx = cid // (ncell * ncell)
        nbr = []
        for oa, ob, oc in offs:
            k = (((cx + oa) % ncell) * ncell + (cy + ob) % ncell) * ncell \
                + (cz + oc) % ncell
            nbr.append(order[starts[k]:starts[k + 1]])
        nbr = np.concatenate(nbr)
        a, b = np.meshgrid(me, nbr, indexing="ij")
        m = a < b
        ii.append(a[m])
        jj.append(b[m])
    if not ii:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def relax(pos, bonds, ref, protein_n, box, rng, sweeps=SWEEPS, noise=NOISE):
    """One frame of evolution: thermal kick, then a few relaxation sweeps."""
    pos = pos + rng.normal(0, noise, pos.shape)
    for _ in range(sweeps):
        f = np.zeros_like(pos)

        # soft repulsion, only for pairs closer than R_MIN
        i, j = cell_pairs(pos % box, box, R_MIN)
        if i.size:
            d = minimum_image(pos[i] - pos[j], box)
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            close = (r < R_MIN) & (r > 1e-9)
            if close.any():
                i, j, d, r = i[close], j[close], d[close], r[close]
                mag = (K_REP * (R_MIN - r) / r)[:, None] * d
                np.add.at(f, i, mag)
                np.add.at(f, j, -mag)

        # harmonic bonds
        bi, bj = bonds[:, 0], bonds[:, 1]
        d = minimum_image(pos[bj] - pos[bi], box)
        r = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-9)
        r0 = np.where(bi < protein_n, BOND_R0, OH)
        fb = (K_BOND * (r - r0) / r)[:, None] * d
        np.add.at(f, bi, fb)
        np.add.at(f, bj, -fb)

        # restrain the protein to its reference fold, so Q(t) stays meaningful
        f[:protein_n] += K_REST * minimum_image(ref - pos[:protein_n], box)

        pos = pos + ETA * f
    return pos % box


HOH_ANGLE = np.radians(104.5)


def orient_waters(pos, oxygens, box, rng, cutoff=0.32):
    """Point one hydrogen of each water at its nearest neighbouring oxygen.

    Without this the hydrogens are randomly oriented and the D-H...A angle
    criterion almost never fires, so the hydrogen-bond analysis would count
    nothing. Real water donates ~2 hydrogen bonds per molecule; aiming the
    first H at the nearest oxygen reproduces that geometry closely enough for
    the analysis to be meaningful.
    """
    o_pos = pos[oxygens]
    n_o = oxygens.size
    i, j = cell_pairs(o_pos, box, cutoff)
    if i.size == 0:
        return pos

    d = minimum_image(o_pos[i] - o_pos[j], box)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    keep = r < cutoff
    i, j, r = i[keep], j[keep], r[keep]
    # consider both directions, then keep each oxygen's nearest partner
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    rr = np.concatenate([r, r])
    order = np.argsort(rr, kind="stable")
    src, dst = src[order], dst[order]
    uniq, first = np.unique(src, return_index=True)
    partner = np.full(n_o, -1, dtype=np.int64)
    partner[uniq] = dst[first]

    has = partner >= 0
    if not has.any():
        return pos
    idx = np.where(has)[0]
    u = minimum_image(o_pos[partner[idx]] - o_pos[idx], box)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)

    # an arbitrary unit vector perpendicular to u, for the second hydrogen
    tmp = rng.normal(0, 1, u.shape)
    perp = tmp - (np.einsum("ij,ij->i", tmp, u))[:, None] * u
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-12)

    o_idx = oxygens[idx]
    pos[o_idx + 1] = (pos[o_idx] + OH * u) % box
    pos[o_idx + 2] = (pos[o_idx] + OH * (np.cos(HOH_ANGLE) * u
                                         + np.sin(HOH_ANGLE) * perp)) % box
    return pos


def generate(size, out_dir, seed=20260810):
    n_protein, n_waters, n_frames = SIZES[size]
    box = float(np.cbrt(n_waters / WATER_DENSITY))
    rng = np.random.default_rng(seed)

    pos, names, types, resids, bonds, donors, acceptors, hydrogens = \
        build_system(n_protein, n_waters, box, rng)
    n_atoms = pos.shape[0]
    ref = pos[:n_protein].copy()

    print(f"[{size}] {n_atoms:,} atoms ({n_protein} protein, {n_waters} waters), "
          f"{n_frames} frames, box {box:.2f} nm")

    print(f"[{size}] equilibrating...")
    for _ in range(40):
        pos = relax(pos, bonds, ref, n_protein, box, rng, sweeps=2, noise=0.0)

    d = Path(out_dir) / size
    d.mkdir(parents=True, exist_ok=True)
    traj = np.lib.format.open_memmap(d / "trajectory.npy", mode="w+",
                                     dtype=np.float32, shape=(n_frames, n_atoms, 3))
    oxygens = np.where(types == 1)[0]
    for frame in range(n_frames):
        pos = relax(pos, bonds, ref, n_protein, box, rng)
        pos = orient_waters(pos, oxygens, box, rng)
        traj[frame] = pos.astype(np.float32)
        if n_frames >= 10 and frame % max(1, n_frames // 10) == 0:
            print(f"  frame {frame:>6d} / {n_frames}", flush=True)
    traj.flush()

    np.savez_compressed(d / "topology.npz", names=names, types=types,
                        resids=resids, bonds=bonds, donors=donors,
                        acceptors=acceptors, hydrogens=hydrogens,
                        box=np.array([box, box, box]), dt_ps=1.0)

    print(f"[{size}] wrote {d}  ({traj.nbytes / 1e6:.1f} MB trajectory)")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=list(SIZES) + ["all"], default="small")
    p.add_argument("--out", default="../data")
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()

    for s in (list(SIZES) if args.size == "all" else [args.size]):
        generate(s, args.out, args.seed)


if __name__ == "__main__":
    main()
