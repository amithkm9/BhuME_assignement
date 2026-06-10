"""Confidence + flag/restraint decisions.

Calibration is the heaviest-weighted axis and it is scored by AUC, so confidence only has
to *rank* corrections by correctness. We turn the aligner's match-quality diagnostics into
a probability that the correction is right (IoU >= 0.5), fit by a small logistic model on
self-supervised synthetic data (see `calibrate.py`) — no example-truth leakage, no
hand-tuned per-village constants.

Three decisions per plot:
  - flagged   : keep the official geometry (low trust, shape/area problem, or no interior min)
  - leave/keep: the plot already sits on its edges → keep official (restraint on controls)
  - corrected : emit the warped geometry with the model's confidence
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Feature order is fixed; the model and feature builder must agree.
FEATURE_NAMES = (
    'prominence',          # depth of the cost well        (higher = sharper lock)
    'margin',              # separation from 2nd basin      (higher = unambiguous)
    'edge_support',        # outline fraction on edges      (higher = real field)
    'improvement_ratio',   # cost drop vs doing nothing     (higher = move was justified)
    'inv_cost',            # 1/(1+chamfer cost)             (higher = tighter fit)
    'area_consistency',    # drawn vs recorded area match   (low = shape error)
    'railed',              # optimum hit search boundary    (1 = suspicious)
    'shift_m',             # size of the correction (m)     (large = riskier; drift ~10-15m)
)


def area_consistency(row) -> float:
    """min/max ratio of drawn area vs recorded total (cultivable + pot-kharaba), in [0,1].

    1.0 = drawn shape matches the record; low = a shape/area error rigid alignment can't fix.
    Returns 1.0 (neutral) when the record is missing — absence isn't evidence of a problem.
    """
    ma = row.get('map_area_sqm')
    rec = row.get('recorded_area_sqm')
    pk = row.get('pot_kharaba_ha')
    if ma is None or rec is None or (isinstance(rec, float) and np.isnan(rec)) or ma <= 0:
        return 1.0
    pk_m2 = pk * 10000.0 if isinstance(pk, (int, float)) and not np.isnan(pk) else 0.0
    total = float(rec) + pk_m2
    if total <= 0:
        return 1.0
    return float(min(ma, total) / max(ma, total))


def build_features(res, row) -> np.ndarray:
    """Feature vector for one aligned plot, in FEATURE_NAMES order."""
    return np.array([
        res.prominence,
        res.margin,
        res.edge_support,
        float(np.clip(res.improvement_ratio, -1.0, 1.0)),
        1.0 / (1.0 + res.cost),
        area_consistency(row),
        1.0 if res.railed else 0.0,
        res.shift_m,
    ], dtype=float)


class LogisticModel:
    """Tiny standardised logistic regression (L-BFGS, L2-regularised)."""

    def __init__(self, names=FEATURE_NAMES):
        self.names = list(names)
        self.w = self.b = self.mu = self.sd = None

    def fit(self, X, y, l2: float = 2.0) -> 'LogisticModel':
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape

        def loss(t):
            w, b = t[:d], t[d]
            z = Xs @ w + b
            return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w * w))

        def grad(t):
            w, b = t[:d], t[d]
            p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
            return np.concatenate([Xs.T @ (p - y) + l2 * w, [np.sum(p - y)]])

        r = minimize(loss, np.zeros(d + 1), jac=grad, method='L-BFGS-B')
        self.w, self.b = r.x[:d], r.x[d]
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        z = ((X - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def weights(self) -> dict:
        return {n: float(w) for n, w in zip(self.names, self.w)}


def heuristic_confidence(res, row) -> float:
    """Transparent fallback when no fitted model is available (a monotone blend in [0,1])."""
    prom = np.clip(res.prominence, 0, 1)
    supp = np.clip(res.edge_support, 0, 1)
    area = area_consistency(row)
    conf = prom * supp * area
    if res.railed:
        conf *= 0.4
    return float(np.clip(conf, 0.0, 1.0))


def decide(res, row, official_geom, model: LogisticModel | None = None,
           flag_below: float = 0.45, keep_imp: float = 0.05, keep_shift_m: float = 4.0,
           area_floor: float = 0.45) -> dict:
    """Turn an AlignResult into a contract decision: status, confidence, geometry, note.

    Gates (in order):
      1. severe area mismatch  -> flag (shape error; a rigid nudge cannot fix it)
      2. railed optimum        -> flag (no interior minimum inside the search window)
      3. already-correct        -> keep official geometry (tiny gain + small shift = restraint)
      4. low confidence         -> flag
      5. otherwise              -> corrected, with the warped geometry and model confidence
    """
    conf = float(model.predict(build_features(res, row))[0]) if model else heuristic_confidence(res, row)
    area = area_consistency(row)

    if area < area_floor:
        return dict(status='flagged', confidence=None, geometry=official_geom,
                    method_note=f'area mismatch (drawn/recorded ratio {area:.2f}) — shape error, not placement')
    if res.railed:
        return dict(status='flagged', confidence=None, geometry=official_geom,
                    method_note='no interior alignment minimum (search railed) — ambiguous')
    if res.improvement_ratio < keep_imp and res.shift_m < keep_shift_m:
        return dict(status='corrected', confidence=round(conf, 3), geometry=official_geom,
                    method_note=f'already on field edges (shift {res.shift_m:.1f}m) — kept official')
    if conf < flag_below:
        return dict(status='flagged', confidence=None, geometry=official_geom,
                    method_note=f'low alignment confidence {conf:.2f}')
    return dict(status='corrected', confidence=round(conf, 3), geometry=res.geometry,
                method_note=f'aligned to field edges (shift {res.shift_m:.1f}m, '
                            f'prom {res.prominence:.2f}, support {res.edge_support:.2f})')
