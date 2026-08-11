#!/usr/bin/env python3
"""
Generate a TESS-like light curve set with planets injected into a known subset.

Each light curve gets a realistic-looking systematic trend, correlated red
noise and white noise; a fraction of them additionally get a box-shaped transit
signal with known period, epoch, duration and depth.

    python generate_data.py --size small
    python generate_data.py --size medium --out ../data

Outputs, per size, into <out>/<size>/ :
    lightcurves.npz   time (shared), flux and flux_err arrays, one row per star
    truth.csv         star_id, has_planet, period, t0, duration, depth

Real TESS data can be substituted once your pipeline works: 2-minute light
curves are at https://archive.stsci.edu/missions-and-data/tess and are most
easily fetched with lightkurve.search_lightcurve(). The confirmed-planet table
at https://exoplanetarchive.ipac.caltech.edu/ is the ground truth there.
"""

import argparse
from pathlib import Path

import numpy as np

SIZES = {
    # name:    (n_stars, n_points, cadence_min, baseline_days)
    "tiny":    (8,     1_500, 30.0, 27.4),
    "small":   (40,    4_000, 10.0, 27.4),
    "medium":  (400,   9_900,  4.0, 27.4),
    "large":   (4_000, 19_700, 2.0, 27.4),
}

PLANET_FRACTION = 0.25


def make_lightcurve(t, rng, inject):
    """One light curve. Returns (flux, flux_err, params or None)."""
    n = t.size

    # --- photometric noise, brighter stars are quieter
    mag = rng.uniform(8.0, 14.0)
    sigma = 1e-4 * 10 ** (0.2 * (mag - 8.0))
    white = rng.normal(0.0, sigma, n)

    # --- correlated red noise (smoothed white noise), the thing that makes
    #     transit detection genuinely hard
    k = max(3, n // 200)
    kernel = np.ones(k) / k
    red = np.convolve(rng.normal(0.0, sigma * 3.0, n + k), kernel, mode="same")[:n]

    # --- slow instrumental trend + a stellar variability sinusoid
    span = t[-1] - t[0]
    trend = (1e-3 * np.sin(2 * np.pi * (t - t[0]) / (span * rng.uniform(0.8, 2.0))
                           + rng.uniform(0, 2 * np.pi)))
    var_p = rng.uniform(0.5, 12.0)
    variability = rng.uniform(0.0, 8e-4) * np.sin(2 * np.pi * t / var_p
                                                  + rng.uniform(0, 2 * np.pi))

    flux = 1.0 + white + red + trend + variability
    params = None

    if inject:
        period = float(rng.uniform(1.0, 9.0))
        t0 = float(t[0] + rng.uniform(0.0, period))
        # duration from a rough transit-scaling relation, then jittered
        duration = float(np.clip(0.06 * period ** (1 / 3) * rng.uniform(0.7, 1.4),
                                 0.03, 0.35))
        depth = float(rng.uniform(3.0, 12.0) * sigma)   # 3-12 sigma per point
        phase = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
        in_transit = phase < 0.5 * duration
        flux[in_transit] -= depth
        params = dict(period=period, t0=t0, duration=duration, depth=depth,
                      n_in_transit=int(in_transit.sum()))

    return flux, np.full(n, sigma), params


def generate(size, out_dir, seed=20260810):
    n_stars, n_points, cadence_min, baseline_days = SIZES[size]
    rng = np.random.default_rng(seed)

    # regular cadence with realistic gaps (downlink gap in the middle)
    t = np.linspace(0.0, baseline_days, n_points)
    keep = ~((t > baseline_days * 0.48) & (t < baseline_days * 0.54))
    t = t[keep]
    n_points = t.size

    flux = np.empty((n_stars, n_points))
    ferr = np.empty((n_stars, n_points))
    rows = []

    print(f"[{size}] {n_stars} stars x {n_points:,} points "
          f"({baseline_days:.1f} d baseline)")
    for s in range(n_stars):
        inject = rng.random() < PLANET_FRACTION
        f, e, p = make_lightcurve(t, rng, inject)
        flux[s], ferr[s] = f, e
        if p:
            rows.append((s, 1, p["period"], p["t0"], p["duration"],
                         p["depth"], p["n_in_transit"]))
        else:
            rows.append((s, 0, np.nan, np.nan, np.nan, np.nan, 0))

    d = Path(out_dir) / size
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "lightcurves.npz", time=t, flux=flux, flux_err=ferr)
    with open(d / "truth.csv", "w") as fh:
        fh.write("star_id,has_planet,period,t0,duration,depth,n_in_transit\n")
        for sid, hp, per, t0, dur, dep, nit in rows:
            fh.write(f"{sid},{hp},{per},{t0},{dur},{dep},{nit}\n")

    n_planets = sum(r[1] for r in rows)
    print(f"[{size}] injected {n_planets} planets into {n_stars} stars")
    print(f"[{size}] wrote {d}")
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
