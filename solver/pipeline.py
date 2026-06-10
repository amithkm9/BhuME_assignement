"""End-to-end: a village bundle -> contract-shaped predictions.

Two stages, justified by the EDA:
  1. estimate the coherent per-village drift (unsupervised, wide translation-only search),
  2. for every plot, refine within a small window around that prior (so dense small plots
     cannot snap onto a wrong neighbour), then decide corrected / flagged / keep-official
     with a confidence from the self-calibrated model.

Nothing is tuned to a particular village: the offset and the confidence model are both
derived per run from the imagery itself.
"""

from __future__ import annotations

import time

import geopandas as gpd
import numpy as np

from bhume.geo import open_imagery

from .align import align_plot, estimate_global_offset
from .calibrate import calibrate_village
from .confidence import decide

THETAS = np.deg2rad(np.arange(-3, 3.01, 1.5))


def solve_village(village, refine_m: float = 11.0, global_search_m: float = 28.0,
                  n_anchors: int = 60, seed: int = 0, verbose: bool = True):
    """Return (predictions GeoDataFrame, report). Plots that error are omitted (not attempted)."""
    plots = village.plots
    report = {}
    t0 = time.time()

    with open_imagery(village.imagery_path) as src:
        gx, gy, kept = estimate_global_offset(src, plots, village.boundaries_path,
                                              search_m=global_search_m, seed=seed)
        report['global_offset_m'] = (round(gx, 2), round(gy, 2))
        report['global_samples'] = kept
        if verbose:
            print(f'[1/3] global offset dx={gx:+.1f} dy={gy:+.1f} m ({kept} samples)')

        model, cal = calibrate_village(src, plots, village.boundaries_path, init=(gx, gy),
                                       search_m=refine_m, n_anchors=n_anchors, seed=seed,
                                       verbose=verbose)
        report['calibration'] = cal
        if verbose:
            print(f'[2/3] calibrated (holdout AUC={cal.get("holdout_auc")})')

        rows = []
        n_corr = n_flag = n_skip = 0
        for pn in plots.index:
            official = plots.loc[pn, 'geometry']
            try:
                r = align_plot(src, official, village.boundaries_path, search_m=refine_m,
                               thetas=THETAS, init_dx_m=gx, init_dy_m=gy)
            except Exception:
                r = None
            if r is None:
                n_skip += 1
                continue
            d = decide(r, plots.loc[pn], official, model=model)
            rows.append({'plot_number': pn, 'status': d['status'],
                         'confidence': d['confidence'], 'method_note': d['method_note'],
                         'geometry': d['geometry']})
            n_corr += d['status'] == 'corrected'
            n_flag += d['status'] == 'flagged'

    preds = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    report.update(n_corrected=n_corr, n_flagged=n_flag, n_skipped=n_skip,
                  seconds=round(time.time() - t0, 1))
    if verbose:
        print(f'[3/3] {n_corr} corrected · {n_flag} flagged · {n_skip} skipped '
              f'[{report["seconds"]}s]')
    return preds, report
