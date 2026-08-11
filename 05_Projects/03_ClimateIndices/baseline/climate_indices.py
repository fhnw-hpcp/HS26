#!/usr/bin/env python3
"""
Baseline ETCCDI climate index computation — deliberately naive, correct, and slow.

This is the code you are asked to make fast. It computes four standard indices
by looping over grid cells and, for the percentile-based one, over days of the
year as well:

  tx90p   fraction of days with tasmax above the calendar-day 90th percentile
          of the reference period, computed with a 5-day window. This is the
          expensive one: one np.percentile call per grid cell per day of year.
  hwdi    heat wave duration index — days in runs of >= 6 consecutive days
          with tasmax more than 5 K above the reference-period daily normal.
          Run-length logic, so it carries state along the time axis.
  gdd     growing degree days, base 10 C. A trivial element-wise reduction,
          included as the contrast case.
  cdd     maximum number of consecutive dry days (pr < 1 mm). Run-length again.

All four are computed per year, giving (year, lat, lon) output fields.

Usage
-----
    python climate_indices.py --size small
    python climate_indices.py --size small --indices tx90p --cells 200

Watch peak memory as well as wall time. The naive version streams one grid cell
at a time and is therefore memory-light and cache-hostile; the obvious
vectorization is memory-hungry. That trade-off is the project.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

DAYS_PER_YEAR = 365
REF_LEN = 30           # reference period length in years, capped at n_years
PCTL = 90              # percentile for tx90p
WINDOW = 5             # calendar-day window, in days (centred)
HW_MIN_RUN = 6         # minimum run length for a heat wave
HW_EXCESS = 5.0        # K above the daily normal
GDD_BASE = 283.15      # 10 C in K
DRY_MM = 1.0           # mm/day below which a day counts as dry


# ----------------------------------------------------------------------------
# Per-cell index kernels — naive on purpose
# ----------------------------------------------------------------------------
def daily_percentile(series, n_years, ref_len, pctl=PCTL, window=WINDOW):
    """Calendar-day percentile of one grid cell over the reference period.

    For each of the 365 days of the year, pool the values from a +/-2 day
    window across all reference years and take the percentile. That is
    365 separate np.percentile calls per grid cell.
    """
    ref = series[:ref_len * DAYS_PER_YEAR].reshape(ref_len, DAYS_PER_YEAR)
    half = window // 2
    out = np.empty(DAYS_PER_YEAR)
    for doy in range(DAYS_PER_YEAR):
        idx = (np.arange(doy - half, doy + half + 1)) % DAYS_PER_YEAR
        out[doy] = np.percentile(ref[:, idx].ravel(), pctl)
    return out


def tx90p_cell(series, n_years, ref_len):
    """Fraction of days per year above the calendar-day 90th percentile."""
    thresh = daily_percentile(series, n_years, ref_len)
    years = series.reshape(n_years, DAYS_PER_YEAR)
    return (years > thresh[None, :]).sum(axis=1) / DAYS_PER_YEAR


def daily_normal(series, ref_len):
    """Calendar-day mean over the reference period."""
    ref = series[:ref_len * DAYS_PER_YEAR].reshape(ref_len, DAYS_PER_YEAR)
    return ref.mean(axis=0)


def hwdi_cell(series, n_years, ref_len):
    """Days in heat-wave runs, per year. Explicit run-length loop."""
    normal = daily_normal(series, ref_len)
    years = series.reshape(n_years, DAYS_PER_YEAR)
    hot = years > (normal[None, :] + HW_EXCESS)
    out = np.zeros(n_years)
    for y in range(n_years):
        run = 0
        total = 0
        for day in range(DAYS_PER_YEAR):
            if hot[y, day]:
                run += 1
            else:
                if run >= HW_MIN_RUN:
                    total += run
                run = 0
        if run >= HW_MIN_RUN:
            total += run
        out[y] = total
    return out


def gdd_cell(series, n_years):
    """Growing degree days, base 10 C. The cheap index."""
    years = series.reshape(n_years, DAYS_PER_YEAR)
    return np.maximum(years - GDD_BASE, 0.0).sum(axis=1)


def cdd_cell(series, n_years):
    """Longest run of consecutive dry days, per year. Explicit run-length loop."""
    years = series.reshape(n_years, DAYS_PER_YEAR)
    dry = years < DRY_MM
    out = np.zeros(n_years)
    for y in range(n_years):
        run = 0
        longest = 0
        for day in range(DAYS_PER_YEAR):
            if dry[y, day]:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
        out[y] = longest
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(path, indices, max_cells=None, progress=True):
    import netCDF4

    ds = netCDF4.Dataset(path, "r")
    n_time = ds.dimensions["time"].size
    n_lat = ds.dimensions["lat"].size
    n_lon = ds.dimensions["lon"].size
    n_years = n_time // DAYS_PER_YEAR
    ref_len = min(REF_LEN, n_years)

    cells = [(i, j) for i in range(n_lat) for j in range(n_lon)]
    if max_cells:
        cells = cells[:max_cells]

    print(f"dataset: {n_time:,} days x {n_lat} x {n_lon} "
          f"({n_years} years, reference {ref_len} years)")
    print(f"computing {indices} over {len(cells):,} grid cells")

    out = {k: np.full((n_years, n_lat, n_lon), np.nan) for k in indices}
    t_io = 0.0
    per_index = {k: 0.0 for k in indices}
    t0 = time.perf_counter()

    for c, (i, j) in enumerate(cells):
        t_a = time.perf_counter()
        # the naive access pattern: one time series at a time, straight through
        # the chunk layout
        tmax = np.asarray(ds["tasmax"][:, i, j], dtype=np.float64)
        pr = (np.asarray(ds["pr"][:, i, j], dtype=np.float64)
              if "cdd" in indices else None)
        t_io += time.perf_counter() - t_a

        if "tx90p" in indices:
            t_a = time.perf_counter()
            out["tx90p"][:, i, j] = tx90p_cell(tmax, n_years, ref_len)
            per_index["tx90p"] += time.perf_counter() - t_a
        if "hwdi" in indices:
            t_a = time.perf_counter()
            out["hwdi"][:, i, j] = hwdi_cell(tmax, n_years, ref_len)
            per_index["hwdi"] += time.perf_counter() - t_a
        if "gdd" in indices:
            t_a = time.perf_counter()
            out["gdd"][:, i, j] = gdd_cell(tmax, n_years)
            per_index["gdd"] += time.perf_counter() - t_a
        if "cdd" in indices:
            t_a = time.perf_counter()
            out["cdd"][:, i, j] = cdd_cell(pr, n_years)
            per_index["cdd"] += time.perf_counter() - t_a

        if progress and len(cells) >= 10 and c % max(1, len(cells) // 10) == 0:
            print(f"  cell {c:>7d} / {len(cells)}", flush=True)

    runtime = time.perf_counter() - t0
    ds.close()

    timing = {"runtime_s": runtime, "io_s": t_io,
              "n_cells": len(cells), "n_years": n_years,
              "ms_per_cell": runtime / len(cells) * 1e3,
              **{f"{k}_s": v for k, v in per_index.items()}}
    return out, timing


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data")
    p.add_argument("--size", default="small")
    p.add_argument("--indices", default="tx90p,hwdi,gdd,cdd")
    p.add_argument("--cells", type=int, default=None,
                   help="only process the first N grid cells (for quick timing)")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    d = Path(args.data) / args.size
    path = d / "era5_like.nc"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run:  python generate_data.py --size {args.size}")

    indices = [s.strip() for s in args.indices.split(",") if s.strip()]
    out, timing = run(path, indices, args.cells, progress=not args.quiet)

    print(f"\nruntime    : {timing['runtime_s']:.2f} s "
          f"({timing['ms_per_cell']:.1f} ms per grid cell)")
    print(f"  I/O      : {timing['io_s']:.2f} s  "
          f"({timing['io_s'] / timing['runtime_s']:.0%})")
    for k in indices:
        print(f"  {k:<8} : {timing[k + '_s']:.2f} s  "
              f"({timing[k + '_s'] / timing['runtime_s']:.0%})")

    print()
    for k in indices:
        v = out[k][np.isfinite(out[k])]
        print(f"{k:<8} mean {v.mean():10.4f}   min {v.min():9.3f}   "
              f"max {v.max():9.3f}   checksum {v.sum():.6e}")

    dest = Path(args.out) if args.out else d / "indices_baseline.npz"
    np.savez_compressed(dest, **out)
    Path(str(dest).replace(".npz", ".json")).write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
