"""PairLogit: Bradley-Terry-style pairwise-logistic within-panel ranker for CC-Search.

The ranker re-orders the facility panel (base H_FIXED + K alternatives) without changing membership.
Training labels (GOOD05) use the historical reference; all 29 base features are GT-free.

Score at inference is r(P) = coef . psi(P) (no intercept; the intercept is fit but discarded).
psi = [standardized 29-D] ++ [within-panel robust-z 29-D] = 58-D.
"""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import constants as K

F0 = ["f0_smaller", "f0_larger", "f0_balance", "f0_ncut", "f0_cutarea", "f0_minnc", "f0_maxnc", "f0_totnc", "f0_nextra", "f0_bc"]
F1 = ["f1_dmin", "f1_dmed", "f1_dmean", "f1_dmax", "f1_dwmean", "f1_within500"]
F2 = ["f2_D", "f2_vi", "f2_nmi", "f2_iou", "f2_jac", "f2_dsf", "f2_dbal", "f2_dncut", "f2_dca", "f2_dtotnc", "f2_dnex", "f2_dbc", "f2_ident"]
FEATS = F0 + F1 + F2   # 29 descriptors


def _chash(p):
    b = p.astype(np.int8).tobytes(); return hashlib.md5(min(b, (1 - p).astype(np.int8).tobytes())).hexdigest()[:16]


def _sncomp(P, side, Ei, Ej, n):
    idx = np.where(P == side)[0]; k = len(idx)
    if k == 0: return 0
    rm = np.full(n, -1); rm[idx] = np.arange(k); em = (P[Ei] == side) & (P[Ej] == side)
    if not em.any(): return k
    return int(connected_components(coo_matrix((np.ones(em.sum()), (rm[Ei[em]], rm[Ej[em]])), shape=(k, k)), directed=False)[0])


def _vinmi(P, Q, m, M):
    ent = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    c = np.array([[m[(P == a) & (Q == b)].sum() for b in (0, 1)] for a in (0, 1)]) / M
    px = c.sum(1); py = c.sum(0); I = 0.0
    for a in (0, 1):
        for b in (0, 1):
            if c[a, b] > 0 and px[a] > 0 and py[b] > 0: I += c[a, b] * np.log2(c[a, b] / (px[a] * py[b]))
    HX, HY = ent(px), ent(py)
    return max(HX + HY - 2 * I, 0.0), (min(max(I / np.sqrt(HX * HY), 0), 1) if HX > 1e-12 and HY > 1e-12 else 1.0)


def _f0_only(Pc, m, M, Ei, Ej, cap, n):
    mS = m[Pc == 0].sum(); mT = M - mS; cut = Pc[Ei] != Pc[Ej]; ncS = _sncomp(Pc, 0, Ei, Ej, n); ncT = _sncomp(Pc, 1, Ei, Ej, n)
    return dict(f0_smaller=min(mS, mT) / M, f0_larger=max(mS, mT) / M, f0_balance=min(mS, mT) / max(mS, mT), f0_ncut=int(cut.sum()), f0_cutarea=float(cap[cut].sum()),
                f0_minnc=min(ncS, ncT), f0_maxnc=max(ncS, ncT), f0_totnc=ncS + ncT, f0_nextra=max(ncS - 1, 0) + max(ncT - 1, 0), f0_bc=int(ncS == 1 and ncT == 1))


def feats29(Pc, m, M, Ei, Ej, cap, edist, P_H, f0H, CsetH, n, feat_order=FEATS):
    cut = Pc[Ei] != Pc[Ej]; f0 = _f0_only(Pc, m, M, Ei, Ej, cap, n)
    if cut.any():
        dd = edist[cut]; aa = cap[cut]
        f1 = dict(f1_dmin=float(dd.min()), f1_dmed=float(np.median(dd)), f1_dmean=float(dd.mean()), f1_dmax=float(dd.max()),
                  f1_dwmean=float((aa * dd).sum() / aa.sum()), f1_within500=float(aa[dd <= 500].sum() / aa.sum()))
    else:
        f1 = dict(f1_dmin=1500.0, f1_dmed=1500.0, f1_dmean=1500.0, f1_dmax=1500.0, f1_dwmean=1500.0, f1_within500=0.0)
    d2 = float((m * (Pc != P_H)).sum()); Dsg = min(d2, M - d2) / M; VI, NMI = _vinmi(Pc, P_H, m, M)
    sA = 0 if m[Pc == 0].sum() <= M / 2 else 1; sB = 0 if m[P_H == 0].sum() <= M / 2 else 1
    AA = Pc == sA; BB = P_H == sB; inter = m[AA & BB].sum(); uni = m[AA | BB].sum(); iou = float(inter / uni) if uni > 0 else 0.0
    Cc = set(np.where(cut)[0].tolist()); jac = len(Cc & CsetH) / len(Cc | CsetH) if (Cc | CsetH) else 1.0
    f2 = dict(f2_D=Dsg, f2_vi=VI, f2_nmi=NMI, f2_iou=iou, f2_jac=float(jac), f2_dsf=f0["f0_smaller"] - f0H["f0_smaller"], f2_dbal=f0["f0_balance"] - f0H["f0_balance"],
              f2_dncut=f0["f0_ncut"] - f0H["f0_ncut"], f2_dca=f0["f0_cutarea"] - f0H["f0_cutarea"], f2_dtotnc=f0["f0_totnc"] - f0H["f0_totnc"], f2_dnex=f0["f0_nextra"] - f0H["f0_nextra"], f2_dbc=f0["f0_bc"] - f0H["f0_bc"], f2_ident=int(_chash(Pc) == _chash(P_H)))
    d = {**f0, **f1, **f2}; return np.array([d[c] for c in feat_order], float), f0


def panel_base29(panel, B):
    """29-D base features for every panel member (panel[0] is the H_FIXED base PH)."""
    Ei, Ej, cap, m, M, edist, n = B["Ei"], B["Ej"], B["cap"], B["m"], B["M"], B["edist"], B["n"]
    PH = panel[0]["P"]; f0H = _f0_only(PH, m, M, Ei, Ej, cap, n); CsetH = set(np.where(PH[Ei] != PH[Ej])[0].tolist())
    X = []
    for mem in panel:
        x, _ = feats29(mem["P"], m, M, Ei, Ej, cap, edist, PH, f0H, CsetH, n)
        X.append(x)
    return np.array(X, float)


def psi(X, med, mu, sd, clip=K.Z_CLIP_PAIRLOGIT):
    """psi = [ (X-mu)/sd ] ++ [ within-panel robust-z clipped ] -> (n, 58)."""
    Xstd = (X - mu) / sd
    medk = np.median(X, 0); madk = np.median(np.abs(X - medk), 0)
    z = np.clip((X - medk) / np.maximum(madk, 1e-6), -clip, clip)
    return np.concatenate([Xstd, z], 1)


def score_panel(X29, model):
    """r(P) = coef . psi(P). Returns per-member scores in panel order."""
    med = np.asarray(model["med29"]); mu = np.asarray(model["mu29"]); sd = np.asarray(model["sd29"]); coef = np.asarray(model["coef58"])
    return psi(X29, med, mu, sd) @ coef


def order_panel(panel, B, model):
    """Return panel-member indices ordered by PairLogit score (descending); base excluded from the ordering."""
    X = panel_base29(panel, B); s = score_panel(X, model)
    alt_idx = [i for i in range(len(panel)) if panel[i].get("is_base", 0) == 0]
    return sorted(alt_idx, key=lambda i: (-s[i], panel[i]["hash"]))


# ---------------------------------------------------------------- training (sklearn logistic on differences)
def train_pairlogit(train_panels, C=K.PAIRLOGIT_C):
    """train_panels: list of dicts {X29:(k,29), mass:(k,), oracle:float}. GOOD05 label = mass within 0.05 of oracle.
    Fits sklearn LogisticRegression on within-panel feature differences of oppositely labeled members.
    Returns a model dict compatible with score_panel/order_panel."""
    from sklearn.linear_model import LogisticRegression
    Xall = np.vstack([p["X29"] for p in train_panels]); med = np.median(Xall, 0)
    Xi = np.where(np.isnan(Xall), med, Xall); mu = Xi.mean(0); sd = Xi.std(0) + 1e-9
    dph, ys, ws = [], [], []
    for p in train_panels:
        X = np.where(np.isnan(p["X29"]), med, p["X29"]); good = (p["mass"] >= p["oracle"] - K.GOOD05_MARGIN).astype(int)
        ph = psi(X, med, mu, sd); k = len(p["mass"]); prs = []
        hashes = p.get("hash", list(range(k)))
        for a in range(k):
            for b in range(a + 1, k):
                if good[a] == good[b]: continue
                A, Bi = (a, b) if str(hashes[a]) < str(hashes[b]) else (b, a)
                prs.append((A, Bi))
        if not prs: continue
        w = 1.0 / len(prs)
        for A, Bi in prs:
            dph.append(ph[A] - ph[Bi]); ys.append(1 if (good[A] == 1 and good[Bi] == 0) else 0); ws.append(w)
    dph = np.array(dph); ys = np.array(ys); ws = np.array(ws)
    if len(np.unique(ys)) < 2:
        coef = np.zeros(dph.shape[1] if len(dph) else 58)
    else:
        coef = LogisticRegression(C=C, max_iter=5000).fit(dph, ys, sample_weight=ws).coef_[0]
    return dict(model="PairLogit", feats29=FEATS, coef58=[float(x) for x in coef],
                med29=[float(x) for x in med], mu29=[float(x) for x in mu], sd29=[float(x) for x in sd])
