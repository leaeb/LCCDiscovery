"""Efficient LCC Discovery via Threshold-Based Candidate Pruning.

Implements Algorithm 2 (Threshold-Based Candidate Discovery) from the paper
section "Efficient Edge Discovery via Threshold-Based Candidate Pruning".

The key idea: a directed pair (x_i, x_j) can only be BH-significant if its
complete rank sum s_(i,j) does not exceed the critical threshold kappa_alpha(c),
where c = |M_i ∩ M_j| is the number of jointly available modalities. A
monotone lower bound s^min_{ij} is accumulated incrementally; once it exceeds
the threshold the pair is permanently pruned without evaluating its full score.

The resulting set of significant mutual pairs is identical to that produced by
an exhaustive pairwise significance test.

This module is fully self-contained (only depends on NumPy).
"""

import numpy as np
from collections import deque
from typing import Dict, List, Set, Tuple

#rank preprocessing

def distance_to_rank_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    """Convert a distance matrix to a rank matrix, robust to NaNs.

    For each row i:
      - the self-distance is set to NaN so it is excluded from ranking,
      - stable ranks are computed among the finite entries only, assigning
        raw ranks 1..n_valid (1 = closest); ties receive the midrank (average
        rank of the tied group) so the expected rank is unbiased,
      - the raw ranks are rescaled onto the common scale {.., n-1} via
        (n-1)/n_valid, so rows with fewer available neighbours (due to
        missingness) are comparable to fully observed rows before being
        summed across modalities; this is the identity when n_valid == n-1,
      - missing entries stay NaN for downstream handling.
    """
    n = dist_matrix.shape[0]
    ranks = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        row = dist_matrix[i, :].astype(float).copy()
        row[i] = np.nan
        mask = np.isfinite(row)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        vals = row[idx]
        order = np.argsort(vals, kind="mergesort")
        sorted_vals = vals[order]
        r_sorted = np.empty(sorted_vals.size, dtype=float)
        pos = 0
        while pos < sorted_vals.size:
            start = pos
            v = sorted_vals[pos]
            while pos + 1 < sorted_vals.size and sorted_vals[pos + 1] == v:
                pos += 1
            end = pos
            midrank = (start + 1 + end + 1) / 2.0
            r_sorted[start : end + 1] = midrank
            pos += 1
        n_valid = idx.size
        r_sorted *= (n - 1) / n_valid
        ranks[i, idx[order]] = r_sorted
    return ranks


def batch_distance_to_rank_matrices(dist_matrices: List[np.ndarray]) -> List[np.ndarray]:
    """Converts a list of distance matrices to a list of rank matrices."""
    return [distance_to_rank_matrix(dm) for dm in dist_matrices]


def exact_proximity_distribution(n: int, m: int) -> np.ndarray:
    """Exact pmf of the sum of m independent discrete-uniform variables on {1,...,n}.

    Returns a 1D array of probabilities for all possible sums from m to m*n.
    """
    pmf = np.ones(n) / n
    for _ in range(m - 1):
        pmf = np.convolve(pmf, np.ones(n) / n)
    return pmf


def compute_p_value_sum(s_ij: float, pmf: np.ndarray, m: int) -> float:
    """One-sided p-value P(Z <= s_ij) for an observed rank sum s_ij."""
    if not np.isfinite(s_ij):
        return 1.0
    idx = int(np.floor(s_ij - m))
    if idx < 0:
        return 0.0
    if idx >= len(pmf):
        return 1.0
    return float(np.sum(pmf[: idx + 1]))


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for a flat array of p-values."""
    n_tests = pvals.size
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
    below = sorted_pvals <= thresholds
    if not np.any(below):
        return np.zeros_like(pvals, dtype=bool)
    max_idx = np.where(below)[0].max()
    sig = np.zeros_like(pvals, dtype=bool)
    sig[sorted_idx[: max_idx + 1]] = True
    return sig


def merge_significant_pairs_into_clusters(C: set, n: int, min_cluster_size: int = 3):
    """Merge significant pairs into clusters via connected components.

    Returns (clusters, noise_set) where clusters is a list of sorted lists of
    instance indices (components of size >= min_cluster_size) and noise_set
    contains all remaining instances.
    """
    adj = {i: set() for i in range(n)}
    for pair in C:
        i, j = tuple(pair)
        adj[i].add(j)
        adj[j].add(i)

    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        comp = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    comp.add(v)
                    q.append(v)
        components.append(comp)

    clusters = [sorted(list(comp)) for comp in components if len(comp) >= min_cluster_size]
    clustered_nodes = set().union(*[set(c) for c in clusters]) if clusters else set()
    all_nodes = set(range(n))
    noise_set = all_nodes - clustered_nodes
    return clusters, noise_set

#preprocessing

def _compute_threshold_table(n_items: int, m: int, alpha: float) -> Dict[int, int]:
    """Compute kappa_alpha(c) for c = 1, ..., m.

    kappa_alpha(c) = max{ s in N : P(Z^c <= s) <= alpha }

    where Z^c is the sum of c independent Uniform({1, ..., n_items}) variables
    (the exact null distribution from Section sec:alg:null of the paper).

    Returns a dict mapping c -> kappa_alpha(c).  If no rank sum satisfies the
    criterion the value is set to c - 1 (one below the support minimum), so
    that the pruning condition s_min > kappa is always triggered immediately.
    """
    table: Dict[int, int] = {}
    for c in range(1, m + 1):
        pmf = exact_proximity_distribution(n_items, c)
        #pmf[k] = P(Z^c = c + k);  cdf[k] = P(Z^c <= c + k)
        cdf = np.cumsum(pmf)
        candidates = np.where(cdf <= alpha)[0]
        if candidates.size == 0:
            #even the minimum possible sum (= c, all ranks = 1) has p-value > alpha.
            table[c] = c - 1
        else:
            table[c] = c + int(candidates[-1])
    return table


def _build_sorted_rank_lists(rank_matrices: List[np.ndarray]) -> List[List[np.ndarray]]:
    """For each source i and modality k build L[i][k]: the sorted array of
    target indices j arranged in ascending rank order (only finite-rank targets,
    diagonal excluded because distance_to_rank_matrix sets it to NaN).

    L[i][k][t] is the target at 0-based position t (i.e. rank t+1) from x_i
    in modality k.
    """
    n = rank_matrices[0].shape[0]
    m = len(rank_matrices)
    L: List[List[np.ndarray]] = []
    for i in range(n):
        L_i: List[np.ndarray] = []
        for k in range(m):
            row = rank_matrices[k][i, :]
            valid = np.flatnonzero(np.isfinite(row))
            if valid.size == 0:
                L_i.append(np.empty(0, dtype=np.intp))
            else:
                order = np.argsort(row[valid], kind="stable")
                L_i.append(valid[order])
        L.append(L_i)
    return L


def _build_c_max_matrix(rank_matrices: List[np.ndarray]) -> np.ndarray:
    """Precompute c_max[i, j] = number of modalities k where rank_matrices[k][i,j]
    is finite (jointly available modality count for each ordered pair).
    """
    n = rank_matrices[0].shape[0]
    c_max = np.zeros((n, n), dtype=np.int32)
    for R in rank_matrices:
        c_max += np.isfinite(R).astype(np.int32)
    np.fill_diagonal(c_max, 0)
    return c_max

#core algorithm

def find_significant_pairs_efficient(
    rank_matrices: List[np.ndarray],
    alpha: float,
    n_items: int,
) -> set:
    """Threshold-based candidate pruning for LCC edge discovery.

    Implements Algorithm 2 from the paper.  Produces exactly the same set of
    significant mutual pairs as the exhaustive procedure but skips candidate
    pairs whose optimistic lower bound on the rank sum already exceeds the
    significance threshold.

    Parameters
    rank_matrices : list of m (n, n) rank matrices (NaN for missing/diagonal)
    alpha         : significance level for row-wise Benjamini-Hochberg FDR
    n_items       : support size of each rank variable (typically n - 1 because
                    self-distances are excluded before ranking)

    Returns
    C : set of frozensets {i, j} — significant mutual (undirected) pairs
    """
    n = rank_matrices[0].shape[0]
    m = len(rank_matrices)

    L = _build_sorted_rank_lists(rank_matrices)
    kappa = _compute_threshold_table(n_items, m, alpha)
    c_max_mat = _build_c_max_matrix(rank_matrices)

    #M[i] = set of modality indices k where L[i][k] is non-empty
    M: List[Set[int]] = [
        {k for k in range(m) if L[i][k].size > 0}
        for i in range(n)
    ]

    #S[i] = targets j for which the directed pair (i -> j) is BH-significant
    S: List[Set[int]] = [set() for _ in range(n)]

    for i in range(n):
        M_i = M[i]
        if not M_i:
            continue

        #stopping criterion uses |M_i| as the tightest safe upper bound on
        #c_max for any pair originating at source i.
        c_i = len(M_i)
        kappa_ci = kappa[c_i]

        #H[j] = {'a': partial rank sum, 'b': modalities accumulated,
        #'c_max': jointly available count, 'seen': set of k already added}
        H: Dict[int, Dict] = {}
        P: Set[int] = set()   #permanently pruned targets
        t_stop = n - 1        #will be shortened by the global stopping criterion

        for t in range(1, n):  #t: 1-based depth / rank position
            for k in M_i:
                rank_list = L[i][k]
                if t > rank_list.size:
                    continue

                j = int(rank_list[t - 1])   #target at rank t from x_i in modality k

                if j in P:
                    continue

                r_k_ij = float(rank_matrices[k][i, j])

                if j not in H:
                    #first encounter: initialise candidate record
                    c_max = int(c_max_mat[i, j])
                    if c_max == 0:
                        P.add(j)
                        continue
                    H[j] = {
                        "a": r_k_ij,
                        "b": 1,
                        "c_max": c_max,
                        "seen": {k},
                    }
                else:
                    #subsequent encounter: update partial sum
                    rec = H[j]
                    rec["a"] += r_k_ij
                    rec["b"] += 1
                    rec["seen"].add(k)

                #lower-bound pruning (eq. s^min_{ij})
                rec = H[j]
                s_min = rec["a"] + (rec["c_max"] - rec["b"])
                if s_min > kappa.get(rec["c_max"], rec["c_max"] - 1):
                    del H[j]
                    P.add(j)

            #global stopping criterion (eq. m(t+1) > kappa_alpha(m))
            #for source i the tightest safe criterion uses c_i = |M_i|:
            #once c_i * (t+1) > kappa_ci, every fully-unseen pair has
            #s^min >= c_i * (t+1) > kappa_ci >= kappa(c^max) for any c^max <= c_i.
            if c_i * (t + 1) > kappa_ci:
                t_stop = t
                break

        #post-traversal: complete scores for partially-seen candidates
        #candidates remaining in H may have accumulated ranks only for the
        #modalities whose sorted list reached them by depth t_stop.  The
        #missing modality ranks (those with r^k_{ij} beyond t_stop in L[i][k])
        #are looked up directly from the rank matrices.
        for j, rec in H.items():
            if rec["b"] < rec["c_max"]:
                for k in range(m):
                    if k not in rec["seen"] and np.isfinite(rank_matrices[k][i, j]):
                        rec["a"] += float(rank_matrices[k][i, j])

        if not H:
            continue

        #row-wise BH with n-1 total tests (eq. 25 of the algorithm)
        #non-candidates (pruned or never seen) are assigned p-value 1.0,
        #which does not change the BH rejection set for candidates in H.
        all_other = np.concatenate(
            [np.arange(0, i, dtype=np.intp), np.arange(i + 1, n, dtype=np.intp)]
        )
        full_pvals = np.ones(n - 1, dtype=float)

        pmf_cache: Dict[int, np.ndarray] = {}
        for idx, jj in enumerate(all_other):
            if jj in H:
                rec = H[int(jj)]
                c_max = rec["c_max"]
                if c_max not in pmf_cache:
                    pmf_cache[c_max] = exact_proximity_distribution(n_items, c_max)
                full_pvals[idx] = compute_p_value_sum(rec["a"], pmf_cache[c_max], c_max)

        sig_mask = benjamini_hochberg(full_pvals, alpha)
        for idx, jj in enumerate(all_other):
            if sig_mask[idx]:
                S[i].add(int(jj))

    #mutual edges and result
    C: set = set()
    for i in range(n):
        for j in S[i]:
            if i in S[j]:
                C.add(frozenset({i, j}))
    return C

#API

def discover_lcc_efficient(
    dist_matrices: List[np.ndarray],
    alpha: float = 0.1,
    min_cluster_size: int = 3,
) -> Tuple[List[List[int]], set]:
    """Efficient LCC discovery using threshold-based candidate pruning.

    Parameters
    ----------
    dist_matrices    : list of m distance matrices, each of shape (n, n).
                       NaN entries indicate missing pairwise distances.
    alpha            : significance level for row-wise Benjamini-Hochberg FDR.
    min_cluster_size : minimum component size to be reported as an LCC
                       (smaller components are treated as noise).

    Returns
    -------
    clusters  : list of sorted lists of instance indices (the LCCs).
    noise_set : set of instance indices not assigned to any LCC.
    """
    rank_matrices = batch_distance_to_rank_matrices(dist_matrices)
    n = rank_matrices[0].shape[0]
    n_items = n - 1   #ranks are over {1, ..., n-1} (self excluded)

    C = find_significant_pairs_efficient(rank_matrices, alpha, n_items)
    clusters, noise_set = merge_significant_pairs_into_clusters(C, n, min_cluster_size)
    return clusters, noise_set
