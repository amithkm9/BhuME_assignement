"""Geometry helpers for the alignment engine.

The kit's `Patch` gives us an RGB crop plus an affine `transform` mapping pixel
(col, row) -> imagery-CRS (x, y) in EPSG:3857. Alignment is done entirely in *patch
pixel space* (that is where image edges live and where shifts are scale-free), then the
chosen transform is pushed back out to lon/lat for the contract output. Doing the search
in pixels — not metres — sidesteps the web-mercator vs UTM scale mismatch entirely,
because every round-trip goes through the actual pixel<->lon/lat mapping.
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform


def transformers(patch):
    """(lonlat->imagery_xy, imagery_xy->lonlat) for a patch's CRS."""
    to_xy = Transformer.from_crs('EPSG:4326', patch.crs, always_xy=True)
    to_ll = Transformer.from_crs(patch.crs, 'EPSG:4326', always_xy=True)
    return to_xy, to_ll


def _affine(transform):
    # rasterio Affine: x = a*col + b*row + c ; y = d*col + e*row + f
    return (transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f)


def px_to_xy(patch, cols, rows):
    """Patch pixel (col, row) centres -> imagery-CRS (x, y)."""
    a, b, c, d, e, f = _affine(patch.transform)
    cc = np.asarray(cols, float) + 0.5
    rr = np.asarray(rows, float) + 0.5
    return a * cc + b * rr + c, d * cc + e * rr + f


def xy_to_px(patch, x, y):
    """Imagery-CRS (x, y) -> patch pixel (col, row) centres."""
    a, b, c, d, e, f = _affine(patch.transform)
    det = a * e - b * d
    x = np.asarray(x, float) - c
    y = np.asarray(y, float) - f
    col = (e * x - b * y) / det - 0.5
    row = (-d * x + a * y) / det - 0.5
    return col, row


def _densify(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Insert points along each segment so consecutive samples are ~`spacing` px apart."""
    out = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        d = np.hypot(*(p1 - p0))
        n = max(int(d / spacing), 1)
        for k in range(n):
            out.append(p0 + (p1 - p0) * (k / n))
    out.append(pts[-1])
    return np.asarray(out)


def _rings(geom: BaseGeometry):
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    for poly in polys:
        yield np.asarray(poly.exterior.coords)
        for ring in poly.interiors:
            yield np.asarray(ring.coords)


def boundary_points_px(patch, geom4326: BaseGeometry, to_xy, spacing_px: float = 1.0) -> np.ndarray:
    """Sample the plot outline as (N, 2) [col, row] points in patch pixel space.

    Every ring (exterior + holes, across multipolygons) is converted to pixels and
    densified to ~1 px spacing, giving the point set the chamfer matcher slides over the
    distance transform.
    """
    chunks = []
    for coords in _rings(geom4326):
        xs, ys = to_xy.transform(coords[:, 0], coords[:, 1])
        cols, rows = xy_to_px(patch, xs, ys)
        chunks.append(_densify(np.column_stack([cols, rows]), spacing_px))
    return np.vstack(chunks)


def warp_polygon(geom4326: BaseGeometry, patch, to_xy, to_ll,
                 dx: float, dy: float, theta: float, cx: float, cy: float) -> BaseGeometry:
    """Apply the pixel-space transform (rotate by `theta` about (cx, cy), then shift by
    (dx, dy)) to a lon/lat polygon and return the warped polygon in lon/lat.

    Rotation sign/convention matches `align.rotate_points`, so the warp reproduces exactly
    the transform the matcher scored.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _warp(xs, ys, z=None):
        X, Y = to_xy.transform(xs, ys)            # lon/lat -> imagery xy
        col, row = xy_to_px(patch, X, Y)          # -> patch pixels
        c0, r0 = col - cx, row - cy
        rc = cos_t * c0 - sin_t * r0 + cx + dx
        rr = sin_t * c0 + cos_t * r0 + cy + dy
        x2, y2 = px_to_xy(patch, rc, rr)          # back to imagery xy
        return to_ll.transform(x2, y2)            # -> lon/lat

    return shp_transform(_warp, geom4326)
