"""Downstream two-way solvers and planar baselines.

  gc    : seeded s-t minimum cut (graph cut)  -- the ContactCut solver
  rw    : nearest-contact random walker (Dirichlet potential)
  msf   : nearest-contact seeded watershed / maximum-spanning forest
  plane : PCA / random planar split through the interface (Geometric / Random baselines)

All produce a length-n leaf-level partition `sl` in {0, 1}. `gc` takes a prebuilt networkx graph G
with 'capacity' edge weights; the others take the edge list (Ei, Ej) and capacities w.
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def gc(G, n, s, t, INF):
    """Seeded s-t min cut pinning node s to source and node t to sink."""
    G.add_edge("S", int(s), capacity=INF); G.add_edge(int(t), "T", capacity=INF)
    _, (sset, _) = nx.minimum_cut(G, "S", "T"); G.remove_node("S"); G.remove_node("T")
    sl = np.ones(n, np.int8); ids = [x for x in sset if not isinstance(x, str)]
    if ids: sl[np.asarray(ids, np.int64)] = 0
    return sl


def gc_sets(G, n, src, snk, INF):
    """Seeded min cut with terminal SETS (used by CC-Search paired-contact orientations)."""
    src = set(int(x) for x in src); snk = set(int(x) for x in snk)
    if not src or not snk or (src & snk): return None
    for s in src: G.add_edge("S", s, capacity=INF)
    for t in snk: G.add_edge(t, "T", capacity=INF)
    try:
        _, (ss, _) = nx.minimum_cut(G, "S", "T")
    finally:
        G.remove_node("S"); G.remove_node("T")
    sl = np.ones(n, np.int8); ids = [x for x in ss if not isinstance(x, str)]
    if ids: sl[np.asarray(ids, np.int64)] = 0
    return sl


def rw(n, Ei, Ej, w, s, t):
    if s == t or n <= 2:
        sl = np.ones(n, np.int8); sl[s] = 0; return sl, "trivial"
    ii = np.concatenate([Ei, Ej]); jj = np.concatenate([Ej, Ei]); ww = np.concatenate([w, w])
    W = sp.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr(); d = np.asarray(W.sum(1)).ravel(); Lap = sp.diags(d) - W
    U = np.array([i for i in range(n) if i != s and i != t])
    if len(U) == 0:
        sl = np.ones(n, np.int8); sl[s] = 0; return sl, "no_unlabeled"
    Luu = (Lap[U][:, U] + sp.eye(len(U)) * 1e-9).tocsc(); Lul = Lap[U][:, [s, t]]
    try:
        pU = spl.spsolve(Luu, -(Lul @ np.array([1.0, 0.0])))
    except Exception:
        pU = np.zeros(len(U))
    p = np.zeros(n); p[s] = 1.0; p[U] = np.nan_to_num(np.asarray(pU).ravel(), nan=0.0)
    sl = (p < 0.5).astype(np.int8); sl[s] = 0; sl[t] = 1
    return sl, "ok"


def msf(n, Ei, Ej, w, s, t):
    order = np.argsort(-w); parent = np.arange(n); label = np.full(n, -1, np.int8); label[s] = 0; label[t] = 1
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    for e in order:
        a = find(int(Ei[e])); b = find(int(Ej[e]))
        if a == b: continue
        la, lb = label[a], label[b]
        if la != -1 and lb != -1 and la != lb: continue
        parent[b] = a
        if label[a] == -1: label[a] = lb
    return np.array([0 if label[find(i)] == 0 else 1 for i in range(n)], np.int8)


def plane(cen, ifloc, normal):
    proj = (cen - ifloc) @ normal; sl = (proj < 0).astype(np.int8)
    if sl.min() == sl.max(): sl = (proj < np.median(proj)).astype(np.int8)
    if sl.min() == sl.max(): sl = (np.arange(len(cen)) % 2).astype(np.int8)
    return sl
