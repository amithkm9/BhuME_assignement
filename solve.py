#!/usr/bin/env python3
"""Run the full method on a village bundle.

    uv run solve.py data/<village_slug>

Estimates the village drift, self-calibrates a confidence model, aligns every plot, writes a
contract-valid predictions.geojson, and self-scores against the example truths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bhume import load, score, write_predictions
from solver.pipeline import solve_village

DEFAULT = 'data/34855_vadnerbhairav_chandavad_nashik'


def main(village_dir: str) -> None:
    v = load(village_dir)
    print(f'== {v.slug} — {len(v.plots)} plots ==')
    preds, report = solve_village(v)

    out = write_predictions(Path(village_dir) / 'predictions.geojson', preds)
    print(f'\nwrote {len(preds)} predictions -> {out}')

    corr = preds[preds['status'] == 'corrected']
    if len(corr):
        c = corr['confidence'].astype(float)
        print(f'confidence: median={c.median():.2f}  p10={c.quantile(.1):.2f}  '
              f'p90={c.quantile(.9):.2f}  (spread is what calibration needs)')

    if v.example_truths is not None:
        print()
        print(score(preds, v))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
