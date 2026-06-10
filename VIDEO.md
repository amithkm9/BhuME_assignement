# 5-minute video script (approach walkthrough)

Screen-record yourself talking through this. Show the code/terminal, not a polished demo —
what you tried, what the data taught you, where it broke, what you'd do next. ~5 min is plenty.

---

### 0:00–0:30 — The problem
- Cadastral plot outlines in Maharashtra's land records sit metres off the real fields — old
  paper maps georeferenced onto satellite imagery, drift where control points are sparse.
- Task: for each plot, decide if the official boundary can be nudged onto the real field and
  where — with a **confidence**, and **flag** the ones you can't place. Submit a *method*.

### 0:30–1:15 — What the data taught me (show EDA)
- Two villages: Vadnerbhairav (large clean fields, 1.2 m/px) and Malatavadi (small dense
  parcels near a town, 0.6 m/px).
- The drift is **mostly one coherent per-village translation** (~10–12 m) **plus a per-plot
  residual** (3–11 m). Confirmed by comparing official vs example-truth centroids.
- `map_area` vs recorded area: ~median 1.0 but a 10–90% spread of 0.7–1.4 → ~20% of plots have a
  real **shape/area** problem rigid alignment can't fix → those should be flagged.
- `boundaries.tif` is a rough edge raster (only 2–5% of pixels lit) — a prior, not truth.

### 1:15–2:30 — The method (show solver/)
- **Edge field**: image gradient OR-fused with `boundaries.tif`, distance-transformed → a smooth
  chamfer cost. Data-adaptive percentile thresholds, so one code path fits both villages.
- **Chamfer alignment**: slide the densified plot outline over the cost surface, search
  translation + small rotation. The *sharpness* of the minimum and *edge-support* tell me how
  trustworthy the lock is.
- **Two stages**: estimate the village offset unsupervised (robust cluster-centre of many fits —
  not from the truths), then refine each plot in a **small window around that prior** so dense
  plots can't wander onto a neighbour.

### 2:30–3:30 — Confidence + restraint (the heaviest-weighted axis)
- Calibration is graded by AUC, so confidence only has to *rank* correctness.
- I **manufacture training data**: confidently-locked plots are pseudo-truths; perturb by known
  offsets, re-align, "did it recover?" is the label, the match diagnostics are the features.
  No example-truth leakage, and it self-calibrates per village.
- Decisions: `corrected` with that confidence, `flagged` (area mismatch / ambiguous / shift too
  large for the plot's size), or kept (already on its edges → restraint on already-correct plots).

### 3:30–4:30 — Results and the debugging arc (the honest part)
- Vadnerbhairav: median IoU 0.61 → **0.87**. Malatavadi: 0.51 → **0.75**. Same code, no tuning.
- Where it broke and how I fixed it:
  - First full run **flagged 74%** of plots → added adaptive search instead of flagging railed ones.
  - Confidence came out **saturated** (no ranking) → switched to regressing continuous IoU.
  - Malatavadi **collapsed to 0.03** — expansion snapped small plots onto neighbours → added a
    **scale-aware shift cap** (a plot can't move more than ~its own radius).

### 4:30–5:00 — Limitations / next
- Malatavadi's global offset is still ~4–5 m off (dense → few clean anchors) → heavy but safe
  flagging; next I'd estimate a local offset field, not one global shift.
- Rigid only — I flag shape errors rather than reshaping; edge-snapping the outline is the next
  accuracy lever, carefully, to avoid overfitting.
