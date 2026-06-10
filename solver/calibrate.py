"""Self-supervised calibration of the confidence model.

We never see ground truth at training time, so we manufacture it: plots that lock sharply
to the imagery are treated as pseudo-truths, perturbed by *known* offsets, and re-aligned.
"Did the aligner find its way back (IoU >= 0.5)?" is a label; the aligner's match diagnostics
are the features. A logistic model fit on this learns what a trustworthy correction looks
like — and it is computed per village, so nothing is hand-tuned to a particular place.

This both *trains* the confidence model and *validates* (held-out AUC) that confidence
actually tracks correctness, which is exactly the heaviest-weighted grading axis.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.stats import spearmanr
from shapely.affinity import translate

from .align import align_plot
from .confidence import RidgeModel, build_features

ACC_IOU = 0.5


def _utm_for(geom) -> str:
    lon = geom.centroid.x
    return f'EPSG:{32600 + int((lon + 180) // 6) + 1}'


def _iou_utm(a, b, utm) -> float:
    ga = gpd.GeoSeries([a], crs='EPSG:4326').to_crs(utm).iloc[0]
    gb = gpd.GeoSeries([b], crs='EPSG:4326').to_crs(utm).iloc[0]
    u = ga.union(gb).area
    return float(ga.intersection(gb).area / u) if u > 0 else 0.0


def _translate_utm(geom, dx_m, dy_m, utm):
    g = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs(utm).iloc[0]
    return gpd.GeoSeries([translate(g, dx_m, dy_m)], crs=utm).to_crs('EPSG:4326').iloc[0]


def auc(scores, labels) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def build_dataset(src, plots, boundaries_path, init=(0.0, 0.0), n_anchors=60, search_m=11.0,
                  perturb_m=(3.0, 6.0, 9.0, 12.0), seed=0, thetas=None):
    """Return (X, y, n_anchors_used) of self-supervised (features, recovered?) pairs.

    Trains in the *inference regime*: every alignment uses the global-offset prior `init`
    (metres) plus a small `search_m` residual window — the same call the pipeline makes per
    plot. A confidently-locked plot's aligned position is its pseudo-truth `star`; we perturb
    its *official* position by known offsets and check whether re-aligning lands back on `star`.
    """
    if thetas is None:
        thetas = np.deg2rad(np.arange(-3, 3.01, 1.5))
    gx, gy = init
    rng = np.random.default_rng(seed)
    order = rng.permutation(list(plots.index))

    X, y, used = [], [], 0
    for pn in order:
        if used >= n_anchors:
            break
        official = plots.loc[pn, 'geometry']
        try:
            a = align_plot(src, official, boundaries_path, search_m=search_m, thetas=thetas,
                           init_dx_m=gx, init_dy_m=gy)
        except Exception:
            continue
        # Only well-locked, unambiguous plots make trustworthy pseudo-truths.
        if a is None or a.railed or a.prominence < 0.55 or a.edge_support < 0.5:
            continue
        star = a.geometry
        utm = _utm_for(official)
        used += 1
        for mag in perturb_m:
            ang = rng.uniform(0, 2 * np.pi)
            start = _translate_utm(official, mag * np.cos(ang), mag * np.sin(ang), utm)
            try:
                r = align_plot(src, start, boundaries_path, search_m=search_m, thetas=thetas,
                               init_dx_m=gx, init_dy_m=gy)
            except Exception:
                continue
            if r is None:
                continue
            X.append(build_features(r, plots.loc[pn]))
            y.append(_iou_utm(r.geometry, star, utm))   # continuous IoU target
    return np.array(X), np.array(y, dtype=float), used


def calibrate_village(src, plots, boundaries_path, init=(0.0, 0.0), search_m=11.0,
                      n_anchors=60, seed=0, verbose=False):
    """Fit and return (LogisticModel, report). Report carries held-out AUC and feature weights."""
    X, y, used = build_dataset(src, plots, boundaries_path, init=init, search_m=search_m,
                               n_anchors=n_anchors, seed=seed)
    report = {'n_samples': len(y), 'n_anchors': used,
              'positive_rate': float((y >= ACC_IOU).mean()) if len(y) else None}
    if len(y) < 20:
        report['note'] = 'insufficient synthetic data — falling back to heuristic confidence'
        return None, report

    # held-out ranking quality on a 70/30 split: does predicted IoU track actual IoU?
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]
    holdout_auc = holdout_spearman = None
    if len(te) >= 5 and len(set((y[tr] >= ACC_IOU).tolist())) == 2:
        pred_te = RidgeModel().fit(X[tr], y[tr]).predict(X[te])
        if len(set((y[te] >= ACC_IOU).tolist())) == 2:
            holdout_auc = auc(pred_te.tolist(), (y[te] >= ACC_IOU).tolist())
        if len(set(pred_te.tolist())) > 1 and len(set(y[te].tolist())) > 1:
            holdout_spearman = round(float(spearmanr(pred_te, y[te]).correlation), 3)
    report['holdout_auc'] = holdout_auc
    report['holdout_spearman'] = holdout_spearman

    model = RidgeModel().fit(X, y)
    report['weights'] = model.weights()
    if verbose:
        print(f'  synthetic: {report["n_samples"]} samples from {used} anchors, '
              f'IoU>=.5 rate={report["positive_rate"]:.2f}, holdout_AUC='
              f'{holdout_auc if holdout_auc is None else round(holdout_auc, 3)}, '
              f'holdout_Spearman={holdout_spearman}')
        print('  feature weights:', {k: round(v, 2) for k, v in report['weights'].items()})
    return model, report
