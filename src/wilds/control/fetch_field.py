"""Fetch a star field from the Gaia catalogue and save it as the simulator's data set.

    python -m wilds.control.fetch_field /Volumes/catalogs/GAIA/DR3 data/sky/m13.npz

The cone search reads files over the share and is slow, so it is done once; the simulated
camera then reads the saved file (``[sky] field_file``). Needs the ``catalogs`` extra. The
default is M13, the field we always use.
"""

import argparse
import time

from .sky import CatsHtmSource, save_field

M13 = (250.4235, 36.4613)  # J2000 degrees: 16h41m41.6s +36d27m41s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gaia_root", help="the catsHTM Gaia folder, e.g. /Volumes/catalogs/GAIA/DR3")
    ap.add_argument("out", help="the .npz file to write")
    ap.add_argument("--ra", type=float, default=M13[0], help="degrees (default: M13)")
    ap.add_argument("--dec", type=float, default=M13[1], help="degrees (default: M13)")
    ap.add_argument("--radius", type=float, default=0.3, help="cone radius, degrees")
    ap.add_argument("--mag-limit", type=float, default=19.0, help="faintest G magnitude kept")
    a = ap.parse_args()
    source = CatsHtmSource(a.gaia_root, mag_limit=a.mag_limit, max_stars=1_000_000)
    t = time.time()
    stars = source.cone(a.ra, a.dec, a.radius)
    save_field(a.out, a.ra, a.dec, a.radius, stars)
    print(f"{len(stars)} stars, G <= {a.mag_limit}, in {time.time() - t:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
