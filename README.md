# BhuMe Boundary Take-Home — Solution

Maharashtra's cadastral plot outlines sit metres off the real fields (an artifact of how old
paper maps were georeferenced onto satellite imagery). **For each plot, this method returns its
best estimate of the true on-the-ground boundary plus a confidence, and flags the ones it can't
place** — submitted as a *method* (`village bundle → predictions.geojson`), not hand-edited maps.

Built on the provided starter kit (`bhume/`, the geospatial plumbing); the method lives in
`solver/`.

## Run it

```bash
uv sync                                                   # one-time: deps into .venv
uv run solve.py data/34855_vadnerbhairav_chandavad_nashik # Vadnerbhairav (Nashik)
uv run solve.py data/12429_malatavadi_chandgad_kolhapur   # Malatavadi (Kolhapur)
```

Each run estimates the village drift, self-calibrates a confidence model, aligns every plot,
writes `data/<village>/predictions.geojson` (contract format), and self-scores against the
public example truths. Download the village bundles from the site's **Get started** page into
`data/` first (the imagery `.tif`s are git-ignored).

## Approach (one method, no per-village tuning)

The drift is dominantly a **coherent per-village translation** (~10–12 m) plus a smaller
**per-plot residual**, with a minority of plots whose *shape/area* is wrong and can't be fixed by
nudging. So:

1. **Edge field** (`solver/edges.py`) — the plot outline is aligned to a fused map of the satellite
   image gradient OR-ed with the rough `boundaries.tif` prior, then a distance transform for a
   smooth chamfer cost. All thresholds are data-adaptive percentiles, so the same code runs on
   both the coarse (1.2 m/px) and fine (0.6 m/px) villages.
2. **Chamfer alignment** (`solver/align.py`) — slide the densified outline over the distance
   transform, searching translation + small rotation; the optimum's sharpness and edge-support
   become confidence signals.
3. **Two stages** — estimate the village offset *unsupervised* (robust cluster-centre of many
   translation-only fits), then refine each plot in a **small window around that prior** so dense
   small parcels can't snap onto a neighbouring field.
4. **Scale-aware** (`solver/confidence.py`) — a correction is capped at ~the plot's own radius;
   railed plots expand only as far as their scale allows. Tight for small parcels, generous for
   large fields, no per-village constant.
5. **Confidence + restraint** — a logistic/ridge model predicts IoU from match-quality features,
   trained on **self-supervised synthetic perturbations** (`solver/calibrate.py`) with *no example-
   truth leakage*. Plots are then `corrected`, `flagged` (area error / ambiguous / implausible
   shift), or kept (already on their edges).

Nothing is tuned to a specific village: the offset and the confidence model are both derived per
run from the imagery itself.

## Results (self-score vs the official starting position)

| Village | Median IoU (mine) | vs official | Calibration |
|---|---|---|---|
| Vadnerbhairav | **0.874** | 0.612 (**+0.262**) | held-out synthetic AUC 0.99 |
| Malatavadi | **0.752** | 0.510 (**+0.242**) | example-truth AUC 1.0 |

These run over the few public truths — a directional check, not the grade. The honest calibration
evidence is the held-out AUC on the self-supervised synthetic set.

## Repo layout

```
solve.py            entry point: village bundle -> predictions.geojson + self-score
solver/
  edges.py          fused edge map + distance transform
  geometry.py       pixel <-> lon/lat, outline sampling, polygon warping
  align.py          chamfer matcher, unsupervised global offset, per-plot refine
  confidence.py     features, ridge confidence model, flag/restraint gates
  calibrate.py      self-supervised synthetic-perturbation calibration
  pipeline.py       end-to-end orchestration
bhume/              provided starter-kit plumbing (I/O, CRS, scoring)
test_align.py       Phase-2 alignment validation
test_confidence.py  Phase-3 calibration + decision validation
transcripts/        AI session logs / web-chat links (how the work was directed)
```

## Limitations (and what I'd do next)

- **Malatavadi's global offset is ~4–5 m off** — dense terrain yields few clean anchors. This
  drives heavy (but safe) flagging there. Next: iterate the offset estimate, or use a coarse
  local offset field instead of one global translation.
- **Rigid only** (translate + small rotation). Plots with genuine shape errors are flagged, not
  reshaped. Edge-snapping the outline could lift IoU further, at the risk of overfitting.
- **Confidence ranking is weak where corrections are nearly all good** (Vadner) — little to rank;
  it carries clear signal where there is real variance (Malatavadi).
