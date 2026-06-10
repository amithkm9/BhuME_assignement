#!/usr/bin/env python3
"""Phase-2 validation: does the alignment engine beat the baseline on the example truths,
and does the unsupervised global offset match the truth-derived shift?"""

import sys
import time

import numpy as np

from bhume import load
from bhume.baseline import _utm_for
from bhume.geo import open_imagery
from solver.align import align_plot, estimate_global_offset

V = sys.argv[1] if len(sys.argv) > 1 else 'data/34855_vadnerbhairav_chandavad_nashik'
THETAS = np.deg2rad(np.arange(-3, 3.01, 1.0))


def iou_utm(a, b, utm):
    import geopandas as gpd
    ga = gpd.GeoSeries([a], crs='EPSG:4326').to_crs(utm).iloc[0]
    gb = gpd.GeoSeries([b], crs='EPSG:4326').to_crs(utm).iloc[0]
    u = ga.union(gb).area
    return ga.intersection(gb).area / u if u > 0 else 0.0


def main():
    v = load(V)
    t = v.example_truths
    utm = _utm_for(t.geometry.iloc[0])
    print(f'{v.slug} — {len(v.plots)} plots, {len(t)} truths\n')

    # 1) unsupervised global offset vs the truth-derived shift
    with open_imagery(v.imagery_path) as src:
        t0 = time.time()
        gdx, gdy, kept = estimate_global_offset(src, v.plots, v.boundaries_path,
                                                n_sample=120, search_m=28.0)
        dt = time.time() - t0
    ou = v.plots.to_crs(utm)
    tu = t.to_crs(utm)
    truth_dx = np.median([tu.loc[i, 'geometry'].centroid.x - ou.loc[i, 'geometry'].centroid.x
                          for i in t.index if i in ou.index])
    truth_dy = np.median([tu.loc[i, 'geometry'].centroid.y - ou.loc[i, 'geometry'].centroid.y
                          for i in t.index if i in ou.index])
    print(f'global offset (unsupervised): dx={gdx:+.1f}m dy={gdy:+.1f}m  '
          f'[{kept} confident samples, {dt:.1f}s]')
    print(f'truth-derived shift (UTM):    dx={truth_dx:+.1f}m dy={truth_dy:+.1f}m  '
          f'(note: engine dy is south-positive, so sign flips vs UTM north)\n')

    # 2) per-plot alignment on the truth plots — IoU vs official and baseline (0.713)
    print(f'{"plot":>8} {"official":>9} {"aligned":>8} {"shift_m":>8} {"prom":>6} '
          f'{"margin":>7} {"edgesup":>8} {"theta":>6}')
    off_ious, new_ious = [], []
    with open_imagery(v.imagery_path) as src:
        for pn in t.index:
            if pn not in v.plots.index:
                continue
            official = v.plots.loc[pn, 'geometry']
            truth = t.loc[pn, 'geometry']
            r = align_plot(src, official, v.boundaries_path, search_m=28.0, thetas=THETAS)
            io = iou_utm(official, truth, utm)
            ia = iou_utm(r.geometry, truth, utm)
            off_ious.append(io)
            new_ious.append(ia)
            print(f'{pn:>8} {io:>9.3f} {ia:>8.3f} {r.shift_m:>8.1f} {r.prominence:>6.2f} '
                  f'{r.margin:>7.2f} {r.edge_support:>8.2f} {np.degrees(r.theta):>6.1f}')

    print(f'\nmedian IoU: official={np.median(off_ious):.3f}  '
          f'aligned={np.median(new_ious):.3f}  baseline=0.713')
    print(f'improved {sum(n > o for n, o in zip(new_ious, off_ious))}/{len(new_ious)} plots')


if __name__ == '__main__':
    main()
