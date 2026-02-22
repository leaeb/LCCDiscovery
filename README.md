# LCC-Discovery

minimal runnable implementation + notebooks for lcc discovery on the paper scenarios.

## contents

- "LCCdiscovery.py": core implementation (numpy-only).
- "miniExample.ipynb": tiny synthetic example.
- "runScenario.ipynb": loads one scenario, plots modalities, runs discovery, plots lccs vs noise.

- "data/scenario1" … "data/scenario6": csv files used by the notebooks.

## quickstart

from the "codeRepo/" folder:

- install deps: "python -m pip install -U numpy matplotlib"
- open and run notebook: "runScenario.ipynb"

in "runScenario.ipynb" (cell 3) set:

- "scenario = 'scenario1'" … "scenario6"
- "alpha", "method" ("exact" or "normal"), "fdr" ("bh" or "by"), "min_component_size"

## notes

- the notebook expects files in "data/<scenario>/" named "modalityX.csv" and "modalityX_dist.csv".
- output is computed as: row-wise tests → fdr control per row → mutual edges → connected components = lccs; all remaining points are noise.
