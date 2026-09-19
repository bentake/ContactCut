"""Per-operation ContactCut pipeline.

build_operation()      -> shared native graph, GT masses, capacities, interface, scorer.
enumerate_candidates() -> single-contact candidates within RADIUS, each with its induced partition Pe and the
                          four structural features (f_bc, f_nextra, f_smaller, f_cutarea) and the H_FIXED scalar sh.
connectivity_tier()    -> T ⊆ C : best available (max f_bc, then min f_nextra).
select_*()             -> Nearest / H_FIXED / CC-Auto contacts.
run_operation()        -> each method's partition + full 3D scores.
"""
import hashlib
import numpy as np
import networkx as nx

from .graph import leaf_graph, contact_centroids
from .scoring import score_common_3d
from .solvers import gc, rw, msf, plane
from . import constants as K

RS_SEED = 20260906  # deterministic RANDOM_SPLIT seed (as in run_classical)

M3D = ["MASS_3D_IOU", "MIN_SIDE_IOU", "SV_3D_IOU", "SIDE_PRECISION", "SIDE_RECALL", "SIDE_F1", "MIN_SIDE_F1",
       "IOU80", "IOU90", "MINSIDE_GE_050", "MINSIDE_GE_080", "MINSIDE_GE_090",
       "SUCCESS@5", "SUCCESS@10", "SUCCESS@15", "VOI_MERGE"]
SHORT = {"MASS_3D_IOU": "MASS", "MIN_SIDE_IOU": "MIN", "SV_3D_IOU": "SV_IOU", "SUCCESS@5": "S5",
         "SUCCESS@10": "S10", "SUCCESS@15": "S15", "VOI_MERGE": "VOI"}


def robustz(v, clip=K.Z_CLIP_AUTO):
    v = np.asarray(v, float); med = np.median(v); mad = np.median(np.abs(v - med))
    if mad == 0 or not np.isfinite(mad): return np.zeros_like(v)
    return np.clip((v - med) / mad, -clip, clip)


def phash(sl):
    p = np.asarray(sl).astype(np.int8)
    return hashlib.md5(min(p.tobytes(), (1 - p).astype(np.int8).tobytes())).hexdigest()


def build_operation(op):
    """Return the shared per-operation state used by every method."""
    res0 = K.res0_for(op["dataset"])
    seg, merged, lo = op["seg"], op["merged"], op["lo"]
    sv, gtl, nid, N = op["sv"], op["gtl"], op["nid"], op["N"]
    leaves, Ei, Ej, fc, cen, vcount, lof = leaf_graph(seg, merged); n = len(leaves)
    lm = {int(v): i for i, v in enumerate(leaves.tolist())}; area = fc.astype(float)
    fg = merged; row_of = np.full(int(nid.max()) + 2, -1, np.int64); row_of[nid] = np.arange(N)
    lif = np.array([lm[int(x)] for x in seg[fg]], np.int64)
    mass = np.bincount(lif, minlength=n).astype(float)                 # GT-free leaf mass (native voxel count)
    gtv = gtl[row_of[sv[fg]]].astype(int)
    cA = np.bincount(lif[gtv == 0], minlength=n).astype(float)         # GT: used only for scoring
    cB = np.bincount(lif[gtv == 1], minlength=n).astype(float)
    GT = np.tile([0, 1], n); Wt = np.empty(2 * n); Wt[0::2] = cA; Wt[1::2] = cB; ONES = np.ones(2 * n, bool)
    def score(sl): return score_common_3d(np.repeat(sl, 2), GT, Wt, present=ONES)
    cen_c = contact_centroids(seg, merged, leaves)
    gv = op["interface_xyz_nm"] / K.CVRES; ifloc = np.array([gv[2] - lo[2], gv[1] - lo[1], gv[0] - lo[0]])
    G = nx.Graph(); G.add_nodes_from(range(n))
    for k in range(len(Ei)):
        wv = max(float(area[k]), 1e-3)
        if G.has_edge(int(Ei[k]), int(Ej[k])): G[int(Ei[k])][int(Ej[k])]["capacity"] += wv
        else: G.add_edge(int(Ei[k]), int(Ej[k]), capacity=wv)
    wdeg = np.array([sum(dd["capacity"] for dd in G[i].values()) for i in range(n)])
    INF = K.INF_MULTIPLIER * (wdeg.sum() + 1)
    wcap = np.array([max(float(a), 1e-3) for a in area])
    return dict(n=n, leaves=leaves, Ei=Ei, Ej=Ej, area=area, cen=cen, mass=mass, G=G, INF=INF, wcap=wcap,
                cen_c=cen_c, ifloc=ifloc, res0=res0, score=score)


def _ncomp(G, sl, side):
    nn = np.where(sl == side)[0]
    return nx.number_connected_components(G.subgraph(nn.tolist())) if len(nn) else 0


def enumerate_candidates(B):
    """Enumerate single-contact candidates within RADIUS and compute their partitions + features."""
    G, n, Ei, Ej, area, mass = B["G"], B["n"], B["Ei"], B["Ej"], B["area"], B["mass"]
    cen_c, ifloc, res0, INF, leaves = B["cen_c"], B["ifloc"], B["res0"], B["INF"], B["leaves"]
    cands = []
    for k in range(len(Ei)):
        u = int(Ei[k]); v = int(Ej[k])
        if u == v: continue
        key = (min(u, v), max(u, v)); cc = cen_c.get(key)
        if cc is None: continue
        di = float(np.linalg.norm((cc - ifloc) * res0))
        if di > K.RADIUS_NM: continue
        sl = gc(G, n, u, v, INF)
        massS = float(mass[sl == 0].sum()); massT = float(mass[sl == 1].sum()); tot = massS + massT + 1e-9
        cutmask = sl[Ei] != sl[Ej]; cutarea = float(area[cutmask].sum())
        ncompS = _ncomp(G, sl, 0); ncompT = _ncomp(G, sl, 1)
        smaller = min(massS, massT) / tot
        nextra = max(ncompS - 1, 0) + max(ncompT - 1, 0)
        bc = int(ncompS == 1 and ncompT == 1)
        sh = (bc * K.HFIX_W_BC - nextra * K.HFIX_W_NEXTRA + smaller * K.HFIX_W_SMALLER - cutarea * K.HFIX_W_CUTAREA)
        cands.append(dict(u=u, v=v, dist=di, sl=sl, f_bc=bc, f_nextra=nextra, f_smaller=smaller,
                          f_cutarea=cutarea, sh=sh, leaf_lo=min(int(leaves[u]), int(leaves[v]))))
    return cands


def connectivity_tier(cands):
    """T = best available connectivity: max f_bc, then min f_nextra within that class."""
    bcmax = max(c["f_bc"] for c in cands)
    inb = [c for c in cands if c["f_bc"] == bcmax]
    nxmin = min(c["f_nextra"] for c in inb)
    return [c for c in cands if c["f_bc"] == bcmax and c["f_nextra"] == nxmin]


def select_nearest(cands):
    return min(cands, key=lambda c: (c["dist"], c["leaf_lo"]))


def select_hfixed(cands):
    return max(cands, key=lambda c: c["sh"])


def select_auto(cands, theta):
    """CC-Auto: rank the connectivity tier by theta @ [z(f_smaller), -z(log1p(f_cutarea))]."""
    tier = connectivity_tier(cands)
    zs = robustz([c["f_smaller"] for c in tier])
    zc = -robustz([np.log1p(max(c["f_cutarea"], 0.0)) for c in tier])
    for i, c in enumerate(tier):
        c["hg"] = float(theta[0] * zs[i] + theta[1] * zc[i])
    return max(tier, key=lambda c: c["hg"]), tier


def run_operation(op, theta, methods=("all",)):
    """Score every 'ours' method for one operation. Returns {method: {metric: value, ...}}."""
    B = build_operation(op); n, Ei, Ej, wcap, cen = B["n"], B["Ei"], B["Ej"], B["wcap"], B["cen"]
    cands = enumerate_candidates(B)
    out = {}
    def emit(name, sl):
        sc = B["score"](sl); r = {"phash": phash(sl), "both_nonempty": int(sl.min() != sl.max())}
        for m in M3D: r[SHORT.get(m, m)] = float(sc[m])
        r["ARI"] = 1.0 - float(sc.get("ARAND", np.nan)); out[name] = r
    if not cands:
        return {"_NO_CANDIDATE": {}}
    nc = select_nearest(cands); hfix = select_hfixed(cands); hg, _tier = select_auto(cands, theta)
    # Selector x solver (Table 2a) + headline partitions (Table 1)
    emit("NC-GC", nc["sl"]); emit("NC-RW", rw(n, Ei, Ej, wcap, nc["u"], nc["v"])[0]); emit("NC-WS_MSF", msf(n, Ei, Ej, wcap, nc["u"], nc["v"]))
    emit("Auto-GC", hg["sl"]); emit("Auto-RW", rw(n, Ei, Ej, wcap, hg["u"], hg["v"])[0]); emit("Auto-WS_MSF", msf(n, Ei, Ej, wcap, hg["u"], hg["v"]))
    emit("CC-Auto", hg["sl"]); emit("H_FIXED", hfix["sl"])
    # planar baselines
    X = cen - cen.mean(0); _, _, vt = np.linalg.svd(X, full_matrices=False); emit("Geometric", plane(cen, B["ifloc"], vt[0]))
    rng = np.random.default_rng(RS_SEED + int(hashlib.md5(str(op.get("operation_id", "0")).encode()).hexdigest()[:8], 16))
    rnrm = rng.standard_normal(3); rnrm /= np.linalg.norm(rnrm) + 1e-12
    emit("Random", plane(cen, B["ifloc"], rnrm)); emit("NO_OP", np.ones(n, np.int8))
    return out
