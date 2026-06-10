#!/usr/bin/env python3
"""Phase-5 honest validation, beyond the handful of public truths.

Uses self-supervised synthetic ground truth (confidently-locked plots as pseudo-truths) to
measure what the public example truths cannot:

  1. Recovery curve  — given a known offset, how often / how well does the full pipeline
                       (global prior + scale-aware refine + decide) put the plot back?
  2. Calibration     — on a held-out split, does predicted confidence rank actual IoU?
                       (AUC, Spearman, and a reliability table).
  3. Restraint       — fed an *already-correct* plot, how often does the pipeline wrongly
                       move it > 5 m? This is the hidden restraint metric, unobservable on the
                       public truths (which contain no already-correct controls).

    uv run validate.py data/<village_slug>
"""

from __future__ import annotations

import sys

import numpy as np

from bhume import load
from bhume.geo import open_imagery
from solver.align import align_plot, estimate_global_offset
from solver.calibrate import _iou_utm, _translate_utm, _utm_for, auc, calibrate_village
from solver.confidence import decide
from solver.pipeline import _align_adaptive

V = sys.argv[1] if len(sys.argv) > 1 else 'data/34855_vadnerbhairav_chandavad_nashik'
THETAS = np.deg2rad(np.arange(-3, 3.01, 1.5))
CONTROL_SHIFT_M = 5.0          # the grader's false-shift threshold
SWEEP_M = (2.0, 5.0, 8.0, 12.0, 16.0, 20.0)
N_ANCHORS = 40


def pick_anchors(src, plots, boundaries, gx, gy, refine_m, n, seed=7):
    """Confidently-locked plots whose aligned position we trust as pseudo-truth."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(list(plots.index))
    anchors = []
    for pn in order:
        if len(anchors) >= n:
            break
        try:
            a = align_plot(src, plots.loc[pn, 'geometry'], boundaries, search_m=refine_m,
                           thetas=THETAS, init_dx_m=gx, init_dy_m=gy)
        except Exception:
            continue
        if a is not None and not a.railed and a.prominence >= 0.6 and a.edge_support >= 0.6:
            anchors.append((pn, a.geometry))
    return anchors


def main():
    v = load(V)
    plots = v.plots
    refine_m, expand_m = 13.0, 24.0
    print(f'== {v.slug} — validation ==')

    with open_imagery(v.imagery_path) as src:
        gx, gy, _ = estimate_global_offset(src, plots, v.boundaries_path, search_m=28.0)
        model, cal = calibrate_village(src, plots, v.boundaries_path, init=(gx, gy),
                                       search_m=refine_m, n_anchors=60)
        print(f'global offset dx={gx:+.1f} dy={gy:+.1f} m · '
              f'calibration holdout AUC={cal.get("holdout_auc")} '
              f'Spearman={cal.get("holdout_spearman")}\n')

        anchors = pick_anchors(src, plots, v.boundaries_path, gx, gy, refine_m, N_ANCHORS)
        print(f'{len(anchors)} synthetic anchors (pseudo-truths)\n')

        # 1) Recovery curve + collect (confidence, IoU) for calibration
        conf_iou = []
        print('1) RECOVERY — perturb a correct plot by a known offset, can the pipeline restore it?')
        print(f'   {"offset":>7} {"medIoU":>7} {"recov>=.5":>9} {"corrected":>9}')
        rng = np.random.default_rng(11)
        for mag in SWEEP_M:
            ious, recov, n_corr = [], 0, 0
            for pn, star in anchors:
                utm = _utm_for(plots.loc[pn, 'geometry'])
                ang = rng.uniform(0, 2 * np.pi)
                start = _translate_utm(star, mag * np.cos(ang), mag * np.sin(ang), utm)
                try:
                    r = _align_adaptive(src, start, plots.loc[pn], v.boundaries_path, gx, gy,
                                        refine_m, expand_m)
                except Exception:
                    r = None
                if r is None:
                    continue
                d = decide(r, plots.loc[pn], start, model=model)
                if d['status'] == 'corrected':
                    n_corr += 1
                    iou = _iou_utm(d['geometry'], star, utm)
                    ious.append(iou)
                    recov += iou >= 0.5
                    if d['confidence'] is not None:
                        conf_iou.append((d['confidence'], iou))
            med = np.median(ious) if ious else float('nan')
            print(f'   {mag:>6.0f}m {med:>7.3f} {recov / max(len(anchors), 1):>9.2f} '
                  f'{n_corr / max(len(anchors), 1):>9.2f}')

        # 2) Calibration: AUC, Spearman, reliability table on the collected pairs
        print('\n2) CALIBRATION — does confidence track accuracy (over all recovery trials)?')
        if len(conf_iou) >= 10:
            cs = [c for c, _ in conf_iou]
            io = [i for _, i in conf_iou]
            a = auc(cs, [i >= 0.5 for i in io])
            from scipy.stats import spearmanr
            sp = spearmanr(cs, io).correlation
            print(f'   n={len(conf_iou)}  AUC(conf, IoU>=.5)={a:.3f}  Spearman(conf,IoU)={sp:+.3f}')
            print(f'   {"conf bin":>12} {"n":>4} {"mean IoU":>9} {"acc>=.5":>8}')
            edges = [0.0, 0.5, 0.7, 0.85, 1.01]
            for lo, hi in zip(edges, edges[1:]):
                grp = [i for c, i in conf_iou if lo <= c < hi]
                if grp:
                    print(f'   [{lo:.2f},{hi if hi <= 1 else 1.0:.2f}) {len(grp):>4} '
                          f'{np.mean(grp):>9.3f} {np.mean([g >= 0.5 for g in grp]):>8.2f}')
        else:
            print('   too few corrected trials to assess')

        # 3) Restraint: feed already-correct plots back in, measure false shifts
        print('\n3) RESTRAINT — feed an already-correct plot, how often is it wrongly moved >5m?')
        shifts, false_shifts, kept = [], 0, 0
        for pn, star in anchors:
            utm = _utm_for(plots.loc[pn, 'geometry'])
            try:
                r = _align_adaptive(src, star, plots.loc[pn], v.boundaries_path, gx, gy,
                                    refine_m, expand_m)
            except Exception:
                r = None
            if r is None:
                continue
            d = decide(r, plots.loc[pn], star, model=model)
            import geopandas as gpd
            g0 = gpd.GeoSeries([star], crs='EPSG:4326').to_crs(utm).iloc[0]
            g1 = gpd.GeoSeries([d['geometry']], crs='EPSG:4326').to_crs(utm).iloc[0]
            shift = g0.centroid.distance(g1.centroid)
            shifts.append(shift)
            false_shifts += shift > CONTROL_SHIFT_M
            kept += d['status'] != 'corrected' or shift <= CONTROL_SHIFT_M
        if shifts:
            print(f'   n={len(shifts)}  median shift={np.median(shifts):.2f}m  '
                  f'false-shift rate (>5m)={false_shifts / len(shifts):.2f}  '
                  f'(lower is better; 0 = never disturbs a correct plot)')


if __name__ == '__main__':
    main()
