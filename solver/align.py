"""The alignment engine: chamfer matching, global-offset estimation, per-plot refine.

Pipeline shape (justified by EDA):
  1. The drift is dominated by a coherent per-village translation (~10-12 m, different per
     village). We estimate it *unsupervised* from the imagery itself — median of many
     per-plot translation-only fits — so it generalises and never touches the example truths.
  2. A substantial per-plot residual remains (3-11 m). From the globally-shifted position we
     run a small-window rigid refine (translation + small rotation) per plot.

Matching is chamfer: slide the densified plot outline over the distance transform of the
edge map; the cost at a shift is the mean distance-to-nearest-edge under the outline. A sharp,
deep minimum means a confident lock; a flat surface means ambiguity (used by Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bhume.geo import patch_for_plot

from . import edges as _edges
from .geometry import boundary_points_px, transformers, warp_polygon


@dataclass
class AlignResult:
    geometry: object              # warped polygon in EPSG:4326
    dx: float                     # chosen shift in patch px (col)
    dy: float                     # chosen shift in patch px (row)
    theta: float                  # chosen rotation (radians)
    cost: float                   # chamfer cost at the optimum (px, lower=better)
    px_per_m: float               # patch resolution, to convert px<->m
    prominence: float             # (median - min) / median of the cost surface  [0,1]
    margin: float                 # 2nd-best-basin separation, normalised
    edge_support: float           # fraction of outline points landing on an edge
    shift_m: float                # magnitude of the chosen translation, in metres
    cost_orig: float              # chamfer cost at the official position (do-nothing)
    railed: bool                  # optimum hit the search-window boundary (no interior min)
    n_points: int = 0
    surface: np.ndarray = field(default=None, repr=False)

    @property
    def improvement_ratio(self) -> float:
        """How much moving reduced the chamfer cost vs doing nothing, in [<=1].

        Near 0 means the plot already sits on edges (restraint: leave it); clearly positive
        means the move found a better-supported position.
        """
        return float((self.cost_orig - self.cost) / (self.cost_orig + 1e-6))


def rotate_points(pts: np.ndarray, theta: float, cx: float, cy: float) -> np.ndarray:
    """Rotate (N,2) [col,row] points by `theta` about (cx, cy). Matches geometry.warp_polygon."""
    if theta == 0.0:
        return pts
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    d = pts - (cx, cy)
    return np.column_stack([
        cos_t * d[:, 0] - sin_t * d[:, 1],
        sin_t * d[:, 0] + cos_t * d[:, 1],
    ]) + (cx, cy)


def _cost_surface(dt: np.ndarray, cols: np.ndarray, rows: np.ndarray, search: int,
                  ox: int = 0, oy: int = 0) -> np.ndarray:
    """Mean chamfer cost over an integer grid in [-search, +search]^2 centred on (ox, oy).

    `(ox, oy)` is the global-offset prior (px); the small window searches residual drift
    around it. Surface cell (iy, ix) is the cost at shift (ox + ix - search, oy + iy - search).
    Vectorised over the outline points; one fancy-index per candidate shift.
    """
    H, W = dt.shape
    rows = rows.astype(int)
    cols = cols.astype(int)
    span = 2 * search + 1
    surf = np.full((span, span), np.inf, np.float32)
    for iy, dy in enumerate(range(-search, search + 1)):
        rr = np.clip(rows + oy + dy, 0, H - 1)
        for ix, dx in enumerate(range(-search, search + 1)):
            cc = np.clip(cols + ox + dx, 0, W - 1)
            surf[iy, ix] = dt[rr, cc].mean()
    return surf


def _surface_stats(surf: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Peak quality: depth of the well (prominence) and separation from the next basin."""
    best = surf[iy, ix]
    med = float(np.median(surf))
    prominence = float((med - best) / (med + 1e-6))
    # second-best outside a small neighbourhood of the optimum
    mask = np.ones_like(surf, bool)
    y0, y1 = max(0, iy - 2), min(surf.shape[0], iy + 3)
    x0, x1 = max(0, ix - 2), min(surf.shape[1], ix + 3)
    mask[y0:y1, x0:x1] = False
    second = float(surf[mask].min()) if mask.any() else med
    margin = float((second - best) / (med + 1e-6))
    return prominence, margin


def align_plot(src, geom4326, boundaries_path, search_m: float, thetas,
               init_dx_m: float = 0.0, init_dy_m: float = 0.0,
               pad_extra_m: float = 12.0, edge_quantile: float = 0.85) -> AlignResult | None:
    """Rigidly align one plot outline to the local edge field.

    `init_dx_m, init_dy_m` are the global-offset prior (metres, patch-pixel convention:
    +x=col/east, +y=row/south); the search is a small `search_m` window around it. Pass
    zeros for an unconstrained search. Returns the warped geometry plus match-quality
    diagnostics (shift measured from the official position, including the prior), or None.
    """
    pad = search_m + max(abs(init_dx_m), abs(init_dy_m)) + pad_extra_m
    patch = patch_for_plot(src, geom4326, pad_m=pad)
    px_per_m = patch.image.shape[1] / (patch.bounds[2] - patch.bounds[0])  # px per imagery-metre
    search = max(1, int(round(search_m * px_per_m)))
    ox = int(round(init_dx_m * px_per_m))
    oy = int(round(init_dy_m * px_per_m))

    field_ = _edges.build(patch, boundaries_path, edge_quantile=edge_quantile)
    to_xy, to_ll = transformers(patch)
    pts = boundary_points_px(patch, geom4326, to_xy)
    if len(pts) == 0:
        return None
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

    # "do nothing" cost: outline at the official position, no rotation/shift
    H0, W0 = field_.dt.shape
    rr0 = np.clip(pts[:, 1].astype(int), 0, H0 - 1)
    cc0 = np.clip(pts[:, 0].astype(int), 0, W0 - 1)
    cost_orig = float(field_.dt[rr0, cc0].mean())

    best = None  # (cost, dx, dy, theta, surface, iy, ix)
    for th in thetas:
        rot = rotate_points(pts, th, cx, cy)
        surf = _cost_surface(field_.dt, rot[:, 0], rot[:, 1], search, ox, oy)
        iy, ix = np.unravel_index(np.argmin(surf), surf.shape)
        cost = float(surf[iy, ix])
        if best is None or cost < best[0]:
            best = (cost, ox + ix - search, oy + iy - search, th, surf, iy, ix)

    cost, dx, dy, theta, surf, iy, ix = best
    prominence, margin = _surface_stats(surf, iy, ix)

    # edge support: fraction of outline points sitting within ~1.5 px of an edge at the optimum
    rot = rotate_points(pts, theta, cx, cy)
    H, W = field_.dt.shape
    rr = np.clip((rot[:, 1] + dy).astype(int), 0, H - 1)
    cc = np.clip((rot[:, 0] + dx).astype(int), 0, W - 1)
    edge_support = float((field_.dt[rr, cc] <= 1.5).mean())

    geom = warp_polygon(geom4326, patch, to_xy, to_ll, dx, dy, theta, cx, cy)
    shift_m = float(np.hypot(dx, dy) / px_per_m)
    # railed = optimum sits on the edge of the residual search window (no interior minimum)
    railed = bool(ix in (0, 2 * search) or iy in (0, 2 * search))

    return AlignResult(
        geometry=geom, dx=float(dx), dy=float(dy), theta=float(theta), cost=cost,
        px_per_m=px_per_m, prominence=prominence, margin=margin,
        edge_support=edge_support, shift_m=shift_m, cost_orig=cost_orig, railed=railed,
        n_points=len(pts), surface=surf,
    )


def estimate_global_offset(src, plots, boundaries_path, n_sample: int = 180,
                           search_m: float = 28.0, min_prominence: float = 0.30,
                           cluster_radius_m: float = 7.0, seed: int = 0):
    """Unsupervised per-village translation: the centre of the dominant drift cluster.

    Samples `n_sample` plots, aligns each translation-only over a wide window, and keeps only
    sharp, non-railed locks (these are the plots whose drift we actually trust). The drift is
    coherent, so the kept offsets form one dominant cluster plus stragglers that snapped to a
    neighbouring field; a plain median is pulled by those stragglers. We instead take the mean
    of the samples within `cluster_radius_m` of the median — the cluster centre. Never uses
    example truths. Returns (dx_m, dy_m) in patch-pixel convention (+x=east, +y=south).
    """
    rng = np.random.default_rng(seed)
    idx = list(plots.index)
    pick = rng.choice(len(idx), size=min(n_sample, len(idx)), replace=False)

    offsets = []
    for i in pick:
        pn = idx[i]
        try:
            r = align_plot(src, plots.loc[pn, 'geometry'], boundaries_path,
                           search_m=search_m, thetas=(0.0,))
        except Exception:
            continue
        if r is None or r.railed or r.prominence < min_prominence:
            continue
        offsets.append((r.dx / r.px_per_m, r.dy / r.px_per_m))

    if len(offsets) < 5:
        raise RuntimeError(f'global offset: only {len(offsets)} confident samples — too few')
    arr = np.array(offsets)
    med = np.median(arr, axis=0)
    keep = np.linalg.norm(arr - med, axis=1) <= cluster_radius_m
    center = arr[keep].mean(axis=0) if keep.sum() >= 5 else med
    return float(center[0]), float(center[1]), int(keep.sum())
