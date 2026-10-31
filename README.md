# Efficient LCC Discovery

Efficient discovery of statistically significant, mutually-close cohorts
("LCCs") across multiple data modalities, using threshold-based candidate
pruning to avoid evaluating every pairwise combination.

Each instance may be observed in several modalities. For every ordered pair
of instances, ranks (based on distance) are aggregated across the jointly
available modalities and tested for significance via a closed-form null
distribution with row-wise Benjamini–Hochberg FDR correction. Mutually
significant pairs are merged into cohorts via connected components. The
pruning strategy in `find_significant_pairs_efficient` produces exactly the
same result as an exhaustive pairwise test, while skipping candidates whose
optimistic lower bound already exceeds the significance threshold.

## Contents

- [`eff_lcc.py`](eff_lcc.py): self-contained implementation (NumPy only)
- [`demo.ipynb`](demo.ipynb): minimal example: generates 2D synthetic data
  with 4 modalities containing one embedded cohort, runs the algorithm, and
  visualizes the result
- [`LICENSE`](LICENSE):  MIT

## Usage

```python
import numpy as np
from eff_lcc import discover_lcc_efficient

# dist_matrices: list of (n, n) distance matrices, one per modality (NaN = missing)
cohorts, noise = discover_lcc_efficient(dist_matrices, alpha=0.1, min_cluster_size=3)
```

## Requirements

- Python 3.9+
- NumPy 
- (matplotlib additionally required to run `demo.ipynb`)

## License

MIT: see [LICENSE](LICENSE).
