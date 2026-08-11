#!/usr/bin/env python3
"""
Generate an ERA5-like daily climate dataset over a European domain.

Synthetic but structurally faithful: a seasonal cycle that varies with
latitude, a warming trend, spatially correlated weather noise, and
precipitation with realistic dry-day frequency and a heavy tail. Written as
NetCDF4 so it behaves like the real thing on disk — chunked, compressed, and
too large to want in memory all at once.

    python generate_data.py --size small
    python generate_data.py --size medium --out ../data

Outputs, per size, into <out>/<size>/ :
    era5_like.nc     tasmax, tasmin, tas (K) and pr (mm/day) on (time, lat, lon)

Real ERA5 can be substituted once your pipeline works: register at
https://cds.climate.copernicus.eu/ and pull daily 2 m temperature and total
precipitation. ARCO-ERA5 is worth studying as a reference Zarr layout.
"""

import argparse
from pathlib import Path

import numpy as np

SIZES = {
    # name:    (n_years, n_lat, n_lon)
    "tiny":    (5,   12,  16),
    "small":   (20,  30,  40),
    "medium":  (45,  60,  80),
    "large":   (45, 200, 280),
}

LAT0, LAT1 = 35.0, 72.0      # Europe
LON0, LON1 = -12.0, 35.0
REF_START, REF_LEN = 0, 30   # reference period for percentiles: first 30 years
DAYS_PER_YEAR = 365


def spatial_noise(shape, rng, smooth=3):
    """Spatially correlated noise, so neighbouring grid cells are not independent."""
    n = rng.normal(0, 1, shape)
    for _ in range(smooth):
        n = (n
             + np.roll(n, 1, axis=-1) + np.roll(n, -1, axis=-1)
             + np.roll(n, 1, axis=-2) + np.roll(n, -1, axis=-2)) / 5.0
    return n / max(n.std(), 1e-9)


def generate(size, out_dir, seed=20260810):
    import netCDF4

    n_years, n_lat, n_lon = SIZES[size]
    n_time = n_years * DAYS_PER_YEAR
    rng = np.random.default_rng(seed)

    lat = np.linspace(LAT0, LAT1, n_lat)
    lon = np.linspace(LON0, LON1, n_lon)
    doy = np.arange(n_time) % DAYS_PER_YEAR
    year = np.arange(n_time) // DAYS_PER_YEAR

    print(f"[{size}] {n_years} years x {n_lat} lat x {n_lon} lon "
          f"= {n_time * n_lat * n_lon / 1e6:.1f} M values per variable")

    d = Path(out_dir) / size
    d.mkdir(parents=True, exist_ok=True)
    path = d / "era5_like.nc"

    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", n_time)
    ds.createDimension("lat", n_lat)
    ds.createDimension("lon", n_lon)

    v_time = ds.createVariable("time", "i4", ("time",))
    v_time.units = "days since 1979-01-01"
    v_time.calendar = "noleap"
    v_time[:] = np.arange(n_time)
    ds.createVariable("lat", "f4", ("lat",))[:] = lat
    ds.createVariable("lon", "f4", ("lon",))[:] = lon
    ds["lat"].units = "degrees_north"
    ds["lon"].units = "degrees_east"

    # chunk along time, which is the layout you get from a naive conversion and
    # emphatically not the layout the percentile indices want
    chunks = (min(365, n_time), min(n_lat, 30), min(n_lon, 40))
    fields = {}
    for name, units, long_name in (
        ("tasmax", "K", "daily maximum near-surface air temperature"),
        ("tasmin", "K", "daily minimum near-surface air temperature"),
        ("tas", "K", "daily mean near-surface air temperature"),
        ("pr", "mm d-1", "precipitation"),
    ):
        v = ds.createVariable(name, "f4", ("time", "lat", "lon"),
                              zlib=True, complevel=4, chunksizes=chunks)
        v.units = units
        v.long_name = long_name
        fields[name] = v

    # latitude sets both the mean temperature and the seasonal amplitude
    base = 288.0 - 0.55 * (lat[:, None] - LAT0)          # K
    amp = 6.0 + 0.22 * (lat[:, None] - LAT0)             # K
    # a mild maritime/continental gradient with longitude
    base = base - 1.5 * np.cos(np.radians(lon[None, :] - LON0))

    block = max(1, min(365, n_time))
    for start in range(0, n_time, block):
        stop = min(start + block, n_time)
        t = slice(start, stop)
        nt = stop - start
        season = np.cos(2 * np.pi * (doy[t] - 200) / DAYS_PER_YEAR)[:, None, None]
        warming = (0.035 * year[t])[:, None, None]       # ~0.35 K/decade

        weather = 3.2 * spatial_noise((nt, n_lat, n_lon), rng)
        # persistence: today looks like yesterday
        for k in range(1, nt):
            weather[k] = 0.72 * weather[k - 1] + 0.69 * weather[k]

        tmean = base[None] - amp[None] * season + warming + weather
        drange = 6.0 + 2.5 * rng.random((nt, n_lat, n_lon))

        fields["tas"][t] = tmean.astype(np.float32)
        fields["tasmax"][t] = (tmean + 0.5 * drange).astype(np.float32)
        fields["tasmin"][t] = (tmean - 0.5 * drange).astype(np.float32)

        # precipitation: wet/dry occurrence then a gamma-ish intensity
        wet_p = 0.30 + 0.12 * season[:, 0, 0][:, None, None]
        wet = rng.random((nt, n_lat, n_lon)) < wet_p
        intensity = rng.gamma(shape=0.7, scale=6.0, size=(nt, n_lat, n_lon))
        fields["pr"][t] = np.where(wet, intensity, 0.0).astype(np.float32)

        print(f"  days {start:>6d} / {n_time}", flush=True)

    ds.reference_period = f"years {REF_START}-{REF_START + REF_LEN - 1}"
    ds.close()

    mb = path.stat().st_size / 1e6
    print(f"[{size}] wrote {path}  ({mb:.1f} MB on disk, "
          f"{n_time * n_lat * n_lon * 4 * 4 / 1e6:.1f} MB uncompressed)")
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
