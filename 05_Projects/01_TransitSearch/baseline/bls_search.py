#!/usr/bin/env python3
"""
Baseline Box Least Squares transit search — deliberately naive, correct, and slow.

This is the code you are asked to make fast. It implements the BLS statistic of
Kovacs, Zucker & Mazeh (2002): for every trial period the light curve is folded
and binned, and then every contiguous block of bins is tested as a candidate
transit. The block search is written as a plain Python double loop, which is
where essentially all of the runtime goes.

For a fold with `nbins` bins and blocks up to `kmax` bins wide, the signal
residue of a block is

    SR = s^2 / (r * (1 - r))

with s the weighted sum of (flux - mean) in the block and r the weighted count.
The largest SR over all blocks is the power at that period.

Usage
-----
    python bls_search.py --size small
    python bls_search.py --size medium --nperiods 3000 --out results.csv

Performance is reported in stars/second and in millions of block evaluations
per second, which is the quantity that actually bounds this kernel.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

SIZES = {
    # name:    n_periods
    "tiny":     400,
    "small":   1_200,
    "medium":  3_000,
    "large":   8_000,
}

NBINS = 200          # phase bins per fold
MIN_DUR_FRAC = 0.01  # shortest trial transit, as a fraction of the period
MAX_DUR_FRAC = 0.12  # longest trial transit


def period_grid(t, n_periods, pmin=0.5, pmax=12.0):
    """Trial periods, uniform in frequency — which is how period space works."""
    fmax, fmin = 1.0 / pmin, 1.0 / pmax
    return 1.0 / np.linspace(fmax, fmin, n_periods)


def flatten(t, y, window_days=0.5):
    """Divide out slow trends with a running median.

    Every transit pipeline does this first: stellar variability and
    instrumental drift produce BLS peaks far stronger than any planet, so
    without it the search is dominated by false positives. A median is used
    rather than a mean so that the transits themselves are not absorbed.

    Naive: one np.median call per data point.
    """
    n = t.size
    half = 0.5 * window_days
    trend = np.empty(n)
    lo = hi = 0
    for i in range(n):
        while t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        trend[i] = np.median(y[lo:hi])
    return y / trend


def bls_periodogram(t, y, dy, periods, nbins=NBINS):
    """BLS power spectrum for one light curve.

    Returns (power, best_period, best_t0, best_duration, best_depth).

    Naive: the fold is vectorized (a scientist would write that much), but the
    block search over (start bin, width) is a Python double loop.
    """
    w = 1.0 / (dy * dy)
    w = w / w.sum()
    x = y - np.sum(w * y)          # weighted-mean-subtracted flux

    kmin = max(1, int(MIN_DUR_FRAC * nbins))
    kmax = max(kmin + 1, int(MAX_DUR_FRAC * nbins))

    power = np.empty(periods.size)
    best = (-1.0, 0.0, 0.0, 0.0, 0.0)   # sr, period, t0, duration, depth
    n_blocks = 0

    for ip in range(periods.size):
        p = periods[ip]

        # ---- fold and bin
        idx = ((t / p) % 1.0 * nbins).astype(np.int64)
        np.clip(idx, 0, nbins - 1, out=idx)
        s = np.bincount(idx, weights=w * x, minlength=nbins)
        r = np.bincount(idx, weights=w, minlength=nbins)

        # ---- block search: every start bin, every width
        best_sr = -1.0
        best_i1 = best_k = 0
        best_s = best_r = 0.0
        for i1 in range(nbins):
            ssum = 0.0
            rsum = 0.0
            for k in range(1, kmax + 1):
                j = i1 + k - 1
                if j >= nbins:
                    j -= nbins
                ssum += s[j]
                rsum += r[j]
                if k < kmin:
                    continue
                if rsum <= 0.0 or rsum >= 1.0:
                    continue
                sr = ssum * ssum / (rsum * (1.0 - rsum))
                if sr > best_sr:
                    best_sr = sr
                    best_i1, best_k = i1, k
                    best_s, best_r = ssum, rsum
        n_blocks += nbins * (kmax - kmin + 1)

        power[ip] = best_sr
        if best_sr > best[0]:
            t0 = (best_i1 + 0.5 * best_k) / nbins * p
            duration = best_k / nbins * p
            depth = -best_s / (best_r * (1.0 - best_r)) if best_r > 0 else 0.0
            best = (best_sr, p, t0, duration, depth)

    return power, best, n_blocks


def sde(power):
    """Signal detection efficiency: the peak, in units of the spectrum's own scatter."""
    med = np.median(power)
    mad = np.median(np.abs(power - med))
    if mad <= 0:
        return 0.0
    return float((power.max() - med) / (1.4826 * mad))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", choices=list(SIZES), default="small")
    p.add_argument("--nperiods", type=int, default=None)
    p.add_argument("--limit", type=int, help="only search the first N stars")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    d = Path(args.data) / args.size
    if not d.exists():
        raise SystemExit(
            f"{d} not found — run:  python generate_data.py --size {args.size}")

    z = np.load(d / "lightcurves.npz")
    t, flux, ferr = z["time"], z["flux"], z["flux_err"]
    if args.limit:
        flux, ferr = flux[:args.limit], ferr[:args.limit]

    n_periods = args.nperiods or SIZES[args.size]
    periods = period_grid(t, n_periods)
    n_stars = flux.shape[0]

    print(f"BLS baseline | {n_stars} stars x {t.size:,} points, "
          f"{n_periods:,} periods, {NBINS} bins")

    rows = []
    total_blocks = 0
    t_flatten = 0.0
    t_start = time.perf_counter()
    for s in range(n_stars):
        t_f0 = time.perf_counter()
        y = flatten(t, flux[s])
        t_flatten += time.perf_counter() - t_f0
        power, best, nb = bls_periodogram(t, y, ferr[s], periods)
        total_blocks += nb
        rows.append((s, best[1], best[2], best[3], best[4], sde(power)))
        if not args.quiet and n_stars >= 5 and s % max(1, n_stars // 10) == 0:
            print(f"  star {s:>5d} / {n_stars}", flush=True)
    runtime = time.perf_counter() - t_start

    timing = {
        "runtime_s": runtime,
        "flatten_s": t_flatten,
        "search_s": runtime - t_flatten,
        "stars_per_s": n_stars / runtime,
        "s_per_star": runtime / n_stars,
        "block_evals": total_blocks,
        "mblocks_per_s": total_blocks / (runtime - t_flatten) / 1e6,
        "n_stars": n_stars, "n_periods": n_periods, "n_points": int(t.size),
    }

    print(f"\nruntime      : {runtime:.2f} s")
    print(f"  detrending : {t_flatten:.2f} s  ({t_flatten / runtime:.0%})")
    print(f"  BLS search : {runtime - t_flatten:.2f} s  "
          f"({(runtime - t_flatten) / runtime:.0%})")
    print(f"per star     : {timing['s_per_star']:.3f} s")
    print(f"block evals  : {total_blocks:,}")
    print(f"M blocks/s   : {timing['mblocks_per_s']:.3f}")

    out = Path(args.out) if args.out else d / "bls_baseline.csv"
    with open(out, "w") as fh:
        fh.write("star_id,period,t0,duration,depth,sde\n")
        for sid, per, t0, dur, dep, s_ in rows:
            fh.write(f"{sid},{per:.10f},{t0:.10f},{dur:.10f},{dep:.10e},{s_:.6f}\n")
    out.with_suffix(".json").write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {out}")
    print("run validate.py to check recovery against the injected truth")


if __name__ == "__main__":
    main()
