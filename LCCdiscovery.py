"""LCC discovery: statistical edge inference from multi-modal distance matrices.

this module implements the core steps described in the paper:
- transform per-modality distance matrices into row-wise rank matrices (midranks)
- aggregate ranks into directed proximity scores s_(i,j)
- compute p-values under a discrete-uniform rank-sum reference null
    (exact by fft-based convolution; optional normal approximation)
- apply row-wise fdr control (bh or by)
- keep mutually significant directed edges and return connected components

the implementation is intentionally self-contained (numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, floor, sqrt
from typing import Iterable, Literal, Sequence

import numpy as np


Method = Literal["exact", "normal"]
FdrMethod = Literal["bh", "by"]


@dataclass(frozen=True)
class LCCConfig:
    alpha: float = 0.05
    method: Method = "exact"
    fdr: FdrMethod = "bh"
    min_component_size: int = 3


@dataclass(frozen=True)
class LCCResult:
    components: list[list[int]]
    noise: list[int]
    mutual_adjacency: np.ndarray  #shape (n, n), bool
    p_values: np.ndarray  #shape (n, n), float (NaN on diagonal / unavailable)
    directed_significant: np.ndarray  #shape (n, n), bool


def discover_lcc(
    distance_matrices: Sequence[np.ndarray],
    config: LCCConfig | None = None,
) -> LCCResult:
    """discover local consensus cohorts (lccs).

    args:
        distance_matrices: sequence of m distance matrices, each shape (n, n).
            matrices may contain nans to indicate missing pairwise distances.
            diagonal entries are ignored.
        config: algorithm configuration.

    returns:
        LCCResult containing components (connected components of the mutual-edge graph),
        noise indices (instances not in a component of size >= min_component_size),
        and intermediate outputs.
    """

    if config is None:
        config = LCCConfig()

    if not distance_matrices:
        raise ValueError("distance_matrices must be a non-empty sequence")

    matrices = [np.asarray(d, dtype=float) for d in distance_matrices]
    n = _validate_square_same_n(matrices)

    rank_mats = [distance_to_row_ranks(d) for d in matrices]
    scores, counts = aggregate_scores(rank_mats)

    if config.method == "exact":
        p_values = pvalues_exact_convolution(scores, counts, n=n)
    elif config.method == "normal":
        p_values = pvalues_normal_approx(scores, counts, n=n)
    else:
        raise ValueError(f"Unknown method: {config.method!r}")

    directed_sig = rowwise_fdr(p_values, alpha=config.alpha, method=config.fdr)
    mutual = directed_sig & directed_sig.T
    np.fill_diagonal(mutual, False)

    components = connected_components(mutual)
    big = [c for c in components if len(c) >= config.min_component_size]

    assigned = np.zeros(n, dtype=bool)
    for comp in big:
        assigned[np.array(comp, dtype=int)] = True
    noise = np.where(~assigned)[0].tolist()

    return LCCResult(
        components=big,
        noise=noise,
        mutual_adjacency=mutual,
        p_values=p_values,
        directed_significant=directed_sig,
    )


def _validate_square_same_n(mats: Sequence[np.ndarray]) -> int:
    first = mats[0]
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("Each distance matrix must be square (n x n)")
    n = first.shape[0]
    for idx, m in enumerate(mats[1:], start=1):
        if m.ndim != 2 or m.shape != (n, n):
            raise ValueError(f"distance_matrices[{idx}] has shape {m.shape}, expected {(n, n)}")
    return n


def distance_to_row_ranks(dist: np.ndarray) -> np.ndarray:
    """convert a distance matrix to a row-wise rank matrix using midranks.

    returns an array R with the same shape as dist where R[i, j] is the rank of
    dist[i, j] among row i (excluding the diagonal). R has nan on the diagonal.
    """

    dist = np.asarray(dist, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square matrix")

    n = dist.shape[0]
    ranks = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        row = dist[i].copy()
        row[i] = np.nan
        finite_mask = np.isfinite(row)
        if not np.any(finite_mask):
            continue

        values = row[finite_mask]
        row_ranks = _midranks(values)

        out_row = np.full(n, np.nan, dtype=float)
        out_row[finite_mask] = row_ranks
        out_row[i] = np.nan
        ranks[i] = out_row

    return ranks


def _midranks(values: np.ndarray) -> np.ndarray:
    """return 1-indexed midranks for a 1d array (no nans)."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1D")

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]

    ranks_sorted = np.empty_like(sorted_x, dtype=float)
    start = 0
    n = len(sorted_x)

    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1

        #positions are 0-indexed; ranks are 1-indexed
        avg_rank = (start + 1 + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def aggregate_scores(rank_mats: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """aggregate per-modality ranks to directed scores and availability counts."""

    if not rank_mats:
        raise ValueError("rank_mats must be non-empty")

    mats = [np.asarray(r, dtype=float) for r in rank_mats]
    n = _validate_square_same_n(mats)

    scores = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)

    for r in mats:
        avail = np.isfinite(r)
        scores[avail] += r[avail]
        counts[avail] += 1

    scores[counts == 0] = np.nan
    np.fill_diagonal(scores, np.nan)
    np.fill_diagonal(counts, 0)
    return scores, counts


def pvalues_exact_convolution(scores: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """compute one-sided p-values p(Z^m <= s) under exact discrete convolution.

    scores can be non-integer due to midranks. since Z^m is integer-valued,
    we use floor(scores) for the cdf threshold.
    """

    scores = np.asarray(scores, dtype=float)
    counts = np.asarray(counts, dtype=int)

    if scores.shape != counts.shape:
        raise ValueError("scores and counts must have the same shape")

    out = np.full_like(scores, np.nan, dtype=float)
    max_m = int(np.nanmax(counts)) if np.any(counts) else 0

    cdf_cache: dict[int, np.ndarray] = {}

    for m in range(1, max_m + 1):
        pmf = _rank_sum_pmf_exact(n=n, m=m)
        cdf = np.cumsum(pmf)
        cdf_cache[m] = cdf

    it = np.nditer(counts, flags=["multi_index"])
    while not it.finished:
        i, j = it.multi_index
        m = int(it[0])
        if m > 0 and i != j and np.isfinite(scores[i, j]):
            s = float(scores[i, j])
            t = floor(s)
            min_sum = m
            max_sum = m * (n - 1)
            if t < min_sum:
                out[i, j] = 0.0
            elif t >= max_sum:
                out[i, j] = 1.0
            else:
                out[i, j] = float(cdf_cache[m][t - min_sum])
        it.iternext()

    np.fill_diagonal(out, np.nan)
    return out


def _rank_sum_pmf_exact(n: int, m: int) -> np.ndarray:
    """pmf of sum of m i.i.d. unif({1,...,n-1}), as a vector over {m,...,m(n-1)}."""

    if n < 2:
        raise ValueError("n must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")

    base = np.full(n - 1, 1.0 / (n - 1), dtype=float)
    pmf = base
    for _ in range(m - 1):
        pmf = _fft_convolve(pmf, base)
        total = pmf.sum()
        if total > 0:
            pmf = pmf / total
    return pmf


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    size = a.size + b.size - 1
    nfft = 1 << (size - 1).bit_length()

    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    out = np.fft.irfft(fa * fb, nfft)[:size]

    out[out < 0] = 0.0
    return out


def pvalues_normal_approx(scores: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """compute one-sided p-values using a normal approximation."""

    scores = np.asarray(scores, dtype=float)
    counts = np.asarray(counts, dtype=int)

    out = np.full_like(scores, np.nan, dtype=float)

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            m = int(counts[i, j])
            if m <= 0 or i == j or not np.isfinite(scores[i, j]):
                continue

            mu = m * (n / 2.0)
            var = m * (n * (n - 2) / 12.0)
            sigma = sqrt(var)

            #continuity correction: Z is integer-valued
            z = (scores[i, j] + 0.5 - mu) / sigma
            out[i, j] = _norm_cdf(z)

    np.fill_diagonal(out, np.nan)
    return out


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def rowwise_fdr(p_values: np.ndarray, alpha: float, method: FdrMethod = "bh") -> np.ndarray:
    """row-wise bh/by fdr control.

    for each row i, tests are all j != i with finite p-values.
    """

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    p = np.asarray(p_values, dtype=float)
    n = p.shape[0]
    sig = np.zeros_like(p, dtype=bool)

    for i in range(n):
        row = p[i].copy()
        row[i] = np.nan
        mask = np.isfinite(row)
        if not np.any(mask):
            continue

        vals = row[mask]
        m = vals.size
        order = np.argsort(vals)
        sorted_p = vals[order]

        if method == "bh":
            denom = 1.0
        elif method == "by":
            denom = float(np.sum(1.0 / np.arange(1, m + 1)))
        else:
            raise ValueError(f"Unknown FDR method: {method!r}")

        thresholds = (alpha / denom) * (np.arange(1, m + 1) / m)
        passed = sorted_p <= thresholds
        if not np.any(passed):
            continue

        k_max = int(np.max(np.where(passed)[0]))
        cutoff = sorted_p[k_max]

        row_sig = np.zeros(n, dtype=bool)
        row_sig[mask] = row[mask] <= cutoff
        sig[i] = row_sig

    np.fill_diagonal(sig, False)
    return sig


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """connected components of an undirected boolean adjacency matrix."""

    adj = np.asarray(adjacency, dtype=bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square")

    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp: list[int] = []

        while stack:
            v = stack.pop()
            comp.append(v)
            neigh = np.flatnonzero(adj[v])
            for u in neigh:
                if not visited[u]:
                    visited[u] = True
                    stack.append(int(u))

        comps.append(sorted(comp))

    return comps
