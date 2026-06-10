#!/usr/bin/env python3
"""Phase-3 validation: self-supervised calibration AUC, then the full decide() pipeline on
the example truths (does confidence track accuracy, and is plot 622 now handled?)."""

import sys
import time

import geopandas as gpd
import numpy as np

from bhume import load
from bhume.baseline import _utm_for
from bhume.geo import open_imagery
from solver.align import align_plot, estimate_global_offset
from solver.calibrate import auc, calibrate_village
from solver.confidence import decide

V = sys.argv[1] if len(sys.argv) > 1 else 'data/34855_vadnerbhairav_chandavad_nashik'
THETAS = np.deg2rad(np.arange(-3, 3.01, 1.5))
REFINE_M = 11.0


def iou_utm(a, b, utm):
    ga = gpd.GeoSeries([a], crs='EPSG:4326').to_crs(utm).iloc[0]
    gb = gpd.GeoSeries([b], crs='EPSG:4326').to_crs(utm).iloc[0]
    u = ga.union(gb).area
    return ga.intersection(gb).area / u if u > 0 else 0.0


def main():
    v = load(V)
    t = v.example_truths
    utm = _utm_for(t.geometry.iloc[0])
    print(f'{v.slug} — {len(v.plots)} plots, {len(t)} truths\n')

    with open_imagery(v.imagery_path) as src:
        t0 = time.time()
        gx, gy, kept = estimate_global_offset(src, v.plots, v.boundaries_path, search_m=28.0)
        print(f'global offset dx={gx:+.1f} dy={gy:+.1f} m ({kept} samples)')
        print('Calibrating (self-supervised, two-stage regime, no truth leakage)...')
        model, rep = calibrate_village(src, v.plots, v.boundaries_path, init=(gx, gy),
                                       search_m=REFINE_M, n_anchors=60, verbose=True)
        print(f'  [{time.time() - t0:.1f}s]\n')

        print(f'{"plot":>8} {"status":>9} {"conf":>5} {"truthIoU":>8} {"officIoU":>8} {"note"}')
        rows = []
        for pn in t.index:
            if pn not in v.plots.index:
                continue
            official = v.plots.loc[pn, 'geometry']
            truth = t.loc[pn, 'geometry']
            r = align_plot(src, official, v.boundaries_path, search_m=REFINE_M, thetas=THETAS,
                           init_dx_m=gx, init_dy_m=gy)
            d = decide(r, v.plots.loc[pn], official, model=model)
            iou_final = iou_utm(d['geometry'], truth, utm)
            iou_off = iou_utm(official, truth, utm)
            rows.append((pn, d, iou_final, iou_off))
            print(f'{pn:>8} {d["status"]:>9} '
                  f'{("-" if d["confidence"] is None else f"{d['confidence']:.2f}"):>5} '
                  f'{iou_final:>8.3f} {iou_off:>8.3f}  {d["method_note"][:48]}')

    corr = [(d['confidence'], iou_f) for _, d, iou_f, _ in rows
            if d['status'] == 'corrected' and d['confidence'] is not None]
    print(f'\nmedian final IoU (corrected+kept) = '
          f'{np.median([iou_f for _, _, iou_f, _ in rows]):.3f}  vs baseline 0.713')
    if len(corr) >= 3:
        a = auc([c for c, _ in corr], [i >= 0.5 for _, i in corr])
        print(f'confidence vs accuracy on truths: AUC={a}  (few plots — synthetic AUC is the real check)')


if __name__ == '__main__':
    main()
