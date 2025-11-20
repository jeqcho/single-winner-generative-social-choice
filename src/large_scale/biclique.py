from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Hashable,
    List,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)
from collections import deque
import math

# ----------------------------
# Public types / data holders
# ----------------------------

CandidateT = TypeVar("CandidateT", bound=Hashable)


@dataclass(frozen=True)
class PVCResult(Generic[CandidateT]):
    """
    Result of computing the Proportional Veto Core (PVC).

    Attributes
    ----------
    core:
        Set of candidates that lie in the proportional veto core.
    r:
        Integer 'r' used in the reduction, satisfying r*n = t*m - gcd(m, n).
    t:
        Integer 't' used in the reduction (t > gcd(m, n) * n), see paper.
    alpha:
        gcd(m, n).
    """
    core: Set[CandidateT]
    r: int
    t: int
    alpha: int


# ----------------------------
# Validation & utilities
# ----------------------------

def _validate_profile(
    profile: Sequence[Sequence[CandidateT]],
) -> Tuple[List[List[CandidateT]], List[CandidateT], int, int]:
    """
    Validate the input profile and return a canonicalized copy.

    Parameters
    ----------
    profile
        Sequence of voters' strict rankings. Each ranking is a sequence of
        distinct hashable candidate IDs (e.g., strings or ints) and must be a
        permutation of the same candidate set.

    Returns
    -------
    (clean_profile, candidates, n, m)
        - clean_profile: list-of-lists copy of the rankings
        - candidates: list of all candidates (as they appear in the first ranking)
        - n: number of voters
        - m: number of candidates

    Raises
    ------
    TypeError
        If inputs are not sequences of sequences of hashable objects.
    ValueError
        If the profile is empty, rankings are empty, rankings have duplicates,
        or voters disagree on the candidate set.
    """
    if not isinstance(profile, Sequence):
        raise TypeError("profile must be a sequence (e.g., list) of rankings")

    n = len(profile)
    if n == 0:
        raise ValueError("profile must contain at least one voter")

    # Coerce rankings to lists and validate hashability + duplicates
    clean: List[List[CandidateT]] = []
    for idx, ranking in enumerate(profile):
        if not isinstance(ranking, Sequence):
            raise TypeError(f"ranking #{idx} must be a sequence of candidates")
        # Check hashability and build list
        rlist: List[CandidateT] = []
        for c in ranking:
            # Hashability is required because we store candidates in sets/maps
            try:
                hash(c)
            except Exception as e:  # pragma: no cover (defensive)
                raise TypeError(f"candidate {c!r} is not hashable") from e
            rlist.append(c)
        if len(rlist) == 0:
            raise ValueError("each ranking must contain at least one candidate")
        if len(rlist) != len(set(rlist)):
            raise ValueError(f"ranking #{idx} contains duplicate candidates")
        clean.append(rlist)

    # All voters must have the same candidate set (strict total orders)
    first_set = set(clean[0])
    m = len(first_set)
    if m == 0:
        raise ValueError("candidate set must be non-empty")
    for idx, rlist in enumerate(clean[1:], start=1):
        if set(rlist) != first_set:
            raise ValueError(
                f"ranking #{idx} has a different candidate set than ranking #0"
            )

    # Canonical candidate order: from the first ballot
    candidates: List[CandidateT] = list(clean[0])
    return clean, candidates, n, m


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclid: returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def _choose_r_t(n: int, m: int) -> Tuple[int, int, int]:
    """
    Choose integers (r, t, alpha) s.t.  r*n = t*m - alpha,  t > alpha*n,  r > 0.

    This follows the construction in the paper’s proof (Theorem 6): first obtain a
    particular solution via Extended Euclid and then shift along the solution family
    to make t sufficiently large and both r, t positive.  We keep r, t as small as
    possible to make the networks sparse.

    Returns
    -------
    (r, t, alpha)
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers")

    alpha = math.gcd(m, n)
    # Find x, y s.t. x*n + y*m = alpha.
    g, x, y = _extended_gcd(n, m)
    assert g == alpha, "extended_gcd should return gcd(n, m)"

    # We want r*n = t*m - alpha  <=>  (-r)*n + t*m = alpha
    r0 = -x
    t0 = y

    # General solution:
    #   r = r0 + k * (m/alpha)
    #   t = t0 + k * (n/alpha)
    step_r = m // alpha
    step_t = n // alpha

    # Choose k to make t > alpha*n and r > 0
    def ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    k_min_t = ceil_div((alpha * n + 1) - t0, step_t)
    k_min_r = 0 if r0 > 0 else ceil_div(1 - r0, step_r)
    k = max(k_min_t, k_min_r)

    r = r0 + k * step_r
    t = t0 + k * step_t
    if not (r > 0 and t > alpha * n):
        # As a final fallback, push k further if needed (should not happen)
        extra = 1 + max(0, (alpha * n + 1 - t) // step_t)
        k += extra
        r = r0 + k * step_r
        t = t0 + k * step_t

    return r, t, alpha


# ----------------------------
# Dinic max-flow implementation
# ----------------------------

class _Dinic:
    """
    Dinic's algorithm for maximum flow with integer capacities.

    We avoid external dependencies and keep this minimal but robust. The graph
    uses adjacency lists with (to, capacity, rev_index) edges.
    """

    def __init__(self, n_vertices: int) -> None:
        if n_vertices <= 1:
            raise ValueError("Dinic graph must have at least 2 vertices")
        self.n = n_vertices
        self.graph: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_vertices)]

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        # forward edge
        self.graph[u].append((v, capacity, len(self.graph[v])))
        # reverse edge (initial capacity 0)
        self.graph[v].append((u, 0, len(self.graph[u]) - 1))

    def _bfs_levels(self, s: int, t: int) -> List[int]:
        level = [-1] * self.n
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for idx, (v, cap, rev) in enumerate(self.graph[u]):
                if cap > 0 and level[v] == -1:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def _dfs_block(self, u: int, t: int, f: int, level: List[int], it: List[int]) -> int:
        if u == t:
            return f
        adj = self.graph[u]
        i = it[u]
        while i < len(adj):
            v, cap, rev = adj[i]
            if cap > 0 and level[u] + 1 == level[v]:
                pushed = self._dfs_block(v, t, min(f, cap), level, it)
                if pushed > 0:
                    # update forward (u->v)
                    adj[i] = (v, cap - pushed, rev)
                    # update reverse (v->u)
                    vr, vcap, vrev = self.graph[v][rev]
                    self.graph[v][rev] = (vr, vcap + pushed, vrev)
                    return pushed
            i += 1
        it[u] = i
        return 0

    def max_flow(self, s: int, t: int) -> int:
        if not (0 <= s < self.n) or not (0 <= t < self.n):
            raise ValueError("s and t must be valid vertex indices")
        flow = 0
        INF = 10 ** 18  # large sentinel
        while True:
            level = self._bfs_levels(s, t)
            if level[t] == -1:
                break
            it = [0] * self.n
            while True:
                pushed = self._dfs_block(s, t, INF, level, it)
                if pushed == 0:
                    break
                flow += pushed
        return flow


# ----------------------------
# Core algorithm
# ----------------------------

def compute_proportional_veto_core(
    profile: Sequence[Sequence[CandidateT]],
) -> PVCResult[CandidateT]:
    """
    Compute the Proportional Veto Core (PVC) for a profile of strict rankings.

    This implementation follows the polynomial-time algorithm described in
    Ianovski & Kondratev (2023), Theorem 6, reducing the blocking test for a
    candidate to a maximum-flow computation in a network constructed from the
    "worse-than-c" relation.  A candidate 'c' is in the PVC iff it is NOT blocked.

    Parameters
    ----------
    profile
        Sequence of voters' strict rankings. Each ranking is a sequence of
        distinct hashable candidate IDs (e.g., strings or ints), and all rankings
        must be permutations of the same candidate set.

    Returns
    -------
    PVCResult
        - core: set of candidates in the proportional veto core
        - r, t, alpha: integers used by the reduction (see the paper)

    Notes
    -----
    Let n be the number of voters, m the number of candidates, and alpha = gcd(m, n).
    Using a particular solution (r, t) with r*n = t*m - alpha and t > alpha*n, the paper
    shows that c is blocked iff the "better-than-c" bipartite graph contains a biclique
    with at least t*m vertices.  Equivalently (via the complement + König’s theorem),
    if F_c is the maximum flow in our network, then
        c is blocked  <=>  F_c <= t*(m - 1) - alpha.
    Therefore, we include c in the core iff F_c > t*(m - 1) - alpha.

    Complexity
    ----------
    For each candidate we solve one max-flow instance with O(n + m) vertices and
    O(n*m) edges in the worst case; using a cubic-time max-flow gives an overall
    O(m * max(n^3, m^3)) bound, matching the paper’s analysis.  In practice this
    Dinic implementation is quite fast for typical sizes.

    References
    ----------
    Egor Ianovski, Aleksei Y. Kondratev (2023). "Computing the proportional veto core".
    See Theorem 6 and its constructive proof. (Algorithmic details and the flow
    construction are drawn from that proof.)  [CITED]  # filecite marker below
    """
    clean, candidates, n, m = _validate_profile(profile)

    # Trivial cases
    if m == 1:
        return PVCResult(core={candidates[0]}, r=1, t=1, alpha=1)

    # Precompute positions for each voter: pos[voter][candidate] -> rank index (0 = best)
    pos: List[Dict[CandidateT, int]] = []
    for rlist in clean:
        pos.append({c: i for i, c in enumerate(rlist)})

    # Compute (r, t, alpha) once for the profile
    r, t, alpha = _choose_r_t(n, m)

    # Threshold for "blocked" per the paper:
    # F_c <= t*(m - 1) - alpha  <=>  c is blocked
    block_threshold = t * (m - 1) - alpha

    core: Set[CandidateT] = set()

    for c in candidates:
        # Build the flow network for candidate c
        # Node indexing:
        #   0                : source S
        #   1..n             : voter nodes
        #   n+1 .. n+(m-1)   : candidate!=c nodes
        #   n+(m-1)+1        : sink T
        sink_index = 1 + n + (m - 1)
        dinic = _Dinic(sink_index + 1)

        S = 0
        T = sink_index

        # Add S -> voter edges (capacity r for each voter)
        for vi in range(n):
            dinic.add_edge(S, 1 + vi, r)

        # Map candidates (except c) to node ids and add candidate -> T edges (capacity t)
        cand_to_node: Dict[CandidateT, int] = {}
        node_cursor = 1 + n
        for d in candidates:
            if d == c:
                continue
            cand_to_node[d] = node_cursor
            dinic.add_edge(node_cursor, T, t)
            node_cursor += 1

        # For each voter, connect to the candidates ranked WORSE than c with "unbounded" cap
        # A safe unbounded sentinel is total incoming capacity per voter (r), but we can use a
        # larger number as well. We use sum of all S->v capacities for safety.
        INF = n * r
        for vi in range(n):
            v_node = 1 + vi
            rank_c = pos[vi][c]
            # All candidates that appear AFTER c in voter vi's order are "worse than c"
            worse_tail = clean[vi][rank_c + 1 :]  # may be empty
            for d in worse_tail:
                d_node = cand_to_node[d]  # d != c guaranteed
                dinic.add_edge(v_node, d_node, INF)

        # Compute max flow F_c for this candidate
        F_c = dinic.max_flow(S, T)

        # Decide blocked vs in-core
        if F_c <= block_threshold:
            # c is blocked -> not in core
            pass
        else:
            core.add(c)

    return PVCResult(core=core, r=r, t=t, alpha=alpha)


def compute_proportional_veto_core_flow(
    profile: Sequence[Sequence[CandidateT]],
) -> PVCResult[CandidateT]:
    """Alias with an explicit name to mirror the brute-force implementation."""

    return compute_proportional_veto_core(profile)