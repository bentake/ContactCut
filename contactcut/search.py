"""CC-Search: bounded paired-contact search + greedy facility-location panel + reachability oracles.

Produces the H_FIXED base, the Best-single / B45-pair / Best-in-K5 reachability oracles, and the
deduplicated facility panel that PairLogit re-orders (pairlogit.py).
"""
import hashlib
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .graph import leaf_graph, contact_centroids
from .scoring import score_common_3d
from . import constants as K


def chash(p):
    b = p.astype(np.int8).tobytes()
    return hashlib.md5(min(b, (1 - p).astype(np.int8).tobytes())).hexdigest()[:16]


def _dist(Pi, Pj, m, M):
    d = float((m * (Pi != Pj)).sum()); return min(d, M - d) / M


def facility(alts, PH, m, M, k):
    """Greedy max-coverage facility location (Nemhauser-Wolsey-Fisher). Base PH seeds coverage."""
    if not alts: return []
    cur = np.maximum(np.array([1 - _dist(PH, u, m, M) for u in alts]), 0.0); sel = []
    while len(sel) < k and len(sel) < len(alts):
        best = None; bg = -1.0; bk = None
        for i in range(len(alts)):
            if i in sel: continue
            s = np.array([1 - _dist(alts[i], u, m, M) for u in alts]); g = float(np.maximum(s, cur).sum() - cur.sum()); key = chash(alts[i])
            if g > bg or (g == bg and (bk is None or key < bk)): bg = g; best = i; bk = key
        if best is None: break
        sel.append(best); cur = np.maximum(cur, np.array([1 - _dist(alts[best], u, m, M) for u in alts]))
    return sel


def build_search(op):
    """Graph + GT masses + contact list + set-seeded solver + f0 + per-edge interface distance."""
    res0 = K.res0_for(op["dataset"])
    seg, merged, lo = op["seg"], op["merged"], op["lo"]
    sv, gtl, nid, N = op["sv"], op["gtl"], op["nid"], op["N"]
    leaves, Ei, Ej, fc, cen, vcount, lof = leaf_graph(seg, merged); n = len(leaves)
    lm = {int(v): i for i, v in enumerate(leaves.tolist())}
    fg = merged; row_of = np.full(int(nid.max()) + 2, -1, np.int64); row_of[nid] = np.arange(N)
    gtv = gtl[row_of[sv[fg]]].astype(int); lif = np.array([lm[int(x)] for x in seg[fg]], np.int64)
    cA = np.bincount(lif[gtv == 0], minlength=n).astype(float); cB = np.bincount(lif[gtv == 1], minlength=n).astype(float)
    m = cA + cB; M = float(m.sum()); GT = np.tile([0, 1], n); Wt = np.empty(2 * n); Wt[0::2] = cA; Wt[1::2] = cB; ONES = np.ones(2 * n, bool)
    cap = np.maximum(fc.astype(float), 1e-3); cen_c = contact_centroids(seg, merged, leaves)
    gv = op["interface_xyz_nm"] / K.CVRES; ifloc = np.array([gv[2] - lo[2], gv[1] - lo[1], gv[0] - lo[0]])
    G = nx.Graph(); G.add_nodes_from(range(n)); agg = {}
    for k in range(len(Ei)):
        a, b = int(Ei[k]), int(Ej[k])
        if a == b: continue
        kk = (min(a, b), max(a, b)); agg[kk] = agg.get(kk, 0.0) + cap[k]
    for (a, b), cc in agg.items(): G.add_edge(a, b, capacity=cc)
    INF = K.INF_MULTIPLIER * (sum(agg.values()) + 1)
    cu, cv, edist = [], [], []
    for k in range(len(Ei)):
        u = int(Ei[k]); v = int(Ej[k])
        if u == v: continue
        c0 = cen_c.get((min(u, v), max(u, v)))
        d = float(np.linalg.norm((c0 - ifloc) * res0)) if c0 is not None else 1500.0
        edist.append(d)
        if c0 is None or d > K.RADIUS_NM: continue
        cu.append(u); cv.append(v)
    edist = np.array(edist, float)
    def score(sl):
        d = score_common_3d(np.repeat(sl, 2), GT, Wt, present=ONES)
        return float(d["MASS_3D_IOU"]), float(d["MIN_SIDE_IOU"]), float(d["SUCCESS@5"]), float(d["SUCCESS@10"]), float(d["SUCCESS@15"])
    def solve(src, snk):
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
    def sncomp(P, side):
        idx = np.where(P == side)[0]; kk = len(idx)
        if kk == 0: return 0
        rm = np.full(n, -1); rm[idx] = np.arange(kk); em = (P[Ei] == side) & (P[Ej] == side)
        if not em.any(): return kk
        return int(connected_components(coo_matrix((np.ones(em.sum()), (rm[Ei[em]], rm[Ej[em]])), shape=(kk, kk)), directed=False)[0])
    def f0(sl):
        mS = m[sl == 0].sum(); mT = M - mS; cut = sl[Ei] != sl[Ej]; ncS = sncomp(sl, 0); ncT = sncomp(sl, 1)
        sf = min(mS, mT) / (mS + mT + 1e-9); nex = max(ncS - 1, 0) + max(ncT - 1, 0); bc = int(ncS == 1 and ncT == 1)
        return bc, nex, sf, float(cap[cut].sum())
    return dict(n=n, Ei=Ei, Ej=Ej, cap=cap, m=m, M=M, cu=np.array(cu), cv=np.array(cv),
                score=score, solve=solve, f0=f0, edist=edist, GT=GT, Wt=Wt, ONES=ONES)


def best_single_oracle(B):
    """Reference-best MASS reachable by any single contact (oracle over C) -- Table 2b 'Best-single'."""
    best = -1.0
    for u, v in zip(B["cu"].tolist(), B["cv"].tolist()):
        sl = B["solve"]({u}, {v})
        if sl is None: continue
        best = max(best, float(B["score"](sl)[0]))
    return best


def hfixed_base(B):
    """H_FIXED single-contact partition PH and its MASS (search-side, argmax sh over contacts)."""
    cu, cv, solve, score, f0 = B["cu"], B["cv"], B["solve"], B["score"], B["f0"]
    best = None
    for u, v in zip(cu.tolist(), cv.tolist()):
        sl = solve({u}, {v})
        if sl is None: continue
        bc, nex, sf, ca = f0(sl)
        sh = bc * K.HFIX_W_BC - nex * K.HFIX_W_NEXTRA + sf * K.HFIX_W_SMALLER - ca * K.HFIX_W_CUTAREA
        if best is None or sh > best[0]: best = (sh, sl.copy())
    if best is None: return None, None
    return best[1], float(score(best[1])[0])


def run_b45(B, PH, QH_mass, b_pair=K.B_PAIR, seed=K.B45_SEED, k_alt=K.K_ALT):
    """Bounded paired-contact search + reachability oracles + facility-K5 panel.

    Returns dict with:
      raw_oracle : B45-pair oracle = max(H_FIXED base MASS, all B45 basin MASS)  (Table 2b 'B45-pair')
      k5_disp    : Best-in-K5      = max(H_FIXED base MASS, facility K alternatives' MASS)
      panel      : list of {hash, P, mass} for the base + facility alternatives (input to PairLogit)
    """
    cu, cv, solve, score, m, M = B["cu"], B["cv"], B["solve"], B["score"], B["m"], B["M"]
    nC = len(cu); hH = chash(PH)
    allpairs = np.array(np.triu_indices(nC, 1)).T if nC >= 2 else np.zeros((0, 2), int)
    if len(allpairs) == 0:
        pairs = []
    elif len(allpairs) <= b_pair:
        pairs = [(int(a), int(b)) for a, b in allpairs]                 # exhaustive ceiling
    else:
        rng = np.random.default_rng(seed); idx = rng.choice(len(allpairs), b_pair, replace=False)
        pairs = [(int(allpairs[k, 0]), int(allpairs[k, 1])) for k in idx]
    basins = {}
    for (i, j) in pairs:
        ui, vi, uj, vj = int(cu[i]), int(cv[i]), int(cu[j]), int(cv[j])
        for orient in (0, 1):
            src = {ui, uj} if orient == 0 else {ui, vj}; snk = {vi, vj} if orient == 0 else {vi, uj}
            sl = solve(src, snk)
            if sl is None: continue
            h = chash(sl)
            if h not in basins: basins[h] = sl
    blist = [(h, sl) for h, sl in basins.items()]; sc = {h: score(sl)[0] for h, sl in blist}
    raw_oracle = max([QH_mass] + [sc[h] for h, _ in blist])
    alts = [sl for h, sl in blist if h != hH]
    sel = facility(alts, PH, m, M, k_alt)
    k5_disp = max([QH_mass] + [float(score(alts[i])[0]) for i in sel])
    panel = [dict(hash=hH, P=PH, mass=QH_mass, is_base=1)]
    for i in sel:
        panel.append(dict(hash=chash(alts[i]), P=alts[i], mass=float(score(alts[i])[0]), is_base=0))
    return dict(nC=nC, n_pairs=len(pairs), n_unique=len(basins), raw_oracle=float(raw_oracle),
                k5_disp=float(k5_disp), base_mass=float(QH_mass), panel=panel, alts=alts)
