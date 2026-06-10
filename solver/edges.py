"""Build the edge / boundary-likelihood map a plot outline is aligned to.

The real field edges live in `imagery.tif` (bunds, tracks, tonal crop boundaries). The
provided `boundaries.tif` is a *rough* pre-detected edge layer — a free prior where it
fires, unreliable where it is thin (EDA: only 2-5% of pixels lit). So we build our own
image gradient and OR-fuse the boundary raster on top, then take the distance transform
so the chamfer cost is smooth (a basin of attraction, not integer-pixel noise).

All thresholds are data-adaptive percentiles, not fixed magnitudes — the same code runs
unchanged on both the coarse (1.2 m/px) and fine (0.6 m/px) villages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy import ndimage


@dataclass
class EdgeField:
    response: np.ndarray   # (H, W) float32 in [0, 1] — fused edge strength
    edges: np.ndarray      # (H, W) bool — thresholded edge pixels
    dt: np.ndarray         # (H, W) float32 — distance (px) to nearest edge


def _gray(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32)
    return 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]


def gradient_magnitude(rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray = ndimage.gaussian_filter(_gray(rgb), sigma)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return np.hypot(gx, gy)


def _norm(a: np.ndarray) -> np.ndarray:
    hi = np.quantile(a, 0.99) + 1e-6
    return np.clip(a / hi, 0.0, 1.0)


def sample_boundaries(boundaries_path, bounds, out_shape, imagery_crs) -> np.ndarray:
    """Read `boundaries.tif` over the patch footprint, resampled to the patch grid.

    `bounds` is (left, bottom, right, top) in the imagery CRS; the boundary raster is
    reprojected-by-bounds if its CRS differs. Returns (H, W) float32.
    """
    left, bottom, right, top = bounds
    with rasterio.open(boundaries_path) as b:
        if str(b.crs) != str(imagery_crs):
            tf = Transformer.from_crs(imagery_crs, b.crs, always_xy=True)
            xs, ys = tf.transform([left, right, left, right], [bottom, bottom, top, top])
            left, right = min(xs), max(xs)
            bottom, top = min(ys), max(ys)
        win = from_bounds(left, bottom, right, top, transform=b.transform)
        arr = b.read(1, window=win, out_shape=out_shape,
                     resampling=Resampling.bilinear, boundless=True, fill_value=0)
    return arr.astype(np.float32)


def build(patch, boundaries_path=None, edge_quantile: float = 0.85,
          boundary_weight: float = 1.0) -> EdgeField:
    """Fuse image gradient with the boundary raster and compute the distance transform."""
    resp = _norm(gradient_magnitude(patch.image))
    if boundaries_path is not None:
        b = sample_boundaries(boundaries_path, patch.bounds, patch.image.shape[:2], patch.crs)
        resp = np.maximum(resp, boundary_weight * _norm(b))

    thr = np.quantile(resp, edge_quantile)
    edges = resp > thr
    if not edges.any():                      # degenerate patch (flat field): fall back
        edges = resp > resp.mean()
    dt = ndimage.distance_transform_edt(~edges).astype(np.float32)
    return EdgeField(response=resp, edges=edges, dt=dt)
