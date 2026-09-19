"""Train CC-Auto's two weights theta by L2-regularized pairwise logistic ranking.

Winner/loser labels come from the historical-reference 3D partition mIoU (MASS); features are GT-free.
C defaults to 1.0 (selected by component-grouped cross-validation); pass a different C to sweep it.
"""
import hashlib
import numpy as np
import pandas as pd

from .pipeline import build_operation, enumerate_candidates, robustz
from . import constants as K


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def build_candidate_table(operations):
    """operations: iterable of loaded op dicts (each must carry 'operation_id'). Returns per-candidate DataFrame
    with the four features and the candidate's MASS (historical-reference 3D partition mIoU)."""
    rows = []
    for op in operations:
        B = build_operation(op)
        for c in enumerate_candidates(B):
            mass = float(B["score"](c["sl"])["MASS_3D_IOU"])
            rows.append(dict(operation_id=str(op["operation_id"]), f_bc=c["f_bc"], f_nextra=c["f_nextra"],
                             f_smaller=c["f_smaller"], f_cutarea=c["f_cutarea"], MASS=mass))
    return pd.DataFrame(rows)


def add_tier_and_phi(cand):
    cand = cand.copy()
    g = cand.groupby("operation_id", sort=False)
    cand["_bc_max"] = g["f_bc"].transform("max")
    in_bc = cand.f_bc.values == cand._bc_max.values
    nx_min = pd.Series(np.where(in_bc, cand.f_nextra.values, np.iinfo(np.int64).max), index=cand.index)\
        .groupby(cand.operation_id, sort=False).transform("min")
    cand["in_tier"] = in_bc & (cand.f_nextra.values == nx_min.values)
    cand["phi_smaller"] = 0.0; cand["phi_cutarea"] = 0.0
    for op, gi in cand[cand.in_tier].groupby("operation_id", sort=False):
        idx = gi.index
        cand.loc[idx, "phi_smaller"] = robustz(gi.f_smaller.values, K.Z_CLIP_AUTO)
        cand.loc[idx, "phi_cutarea"] = -robustz(np.log1p(np.maximum(gi.f_cutarea.values, 0.0)), K.Z_CLIP_AUTO)
    return cand.drop(columns=["_bc_max"])


def make_pairs(tier):
    tier = tier.reset_index(drop=True); win, lose, wt = [], [], []
    for op, g in tier.groupby("operation_id", sort=False):
        pos = g.index.values; m = g.MASS.values; prs = []
        for ia in range(len(pos)):
            for ib in range(ia + 1, len(pos)):
                if abs(m[ia] - m[ib]) >= K.PAIR_MARGIN: prs.append((pos[ia], pos[ib]))
        if not prs: continue
        if len(prs) > K.PAIR_CAP:
            rng = np.random.default_rng(K.PAIR_SEED_BASE + int(hashlib.md5(str(op).encode()).hexdigest()[:8], 16))
            keep = sorted(rng.choice(len(prs), K.PAIR_CAP, replace=False)); prs = [prs[i] for i in keep]
        w = 1.0 / len(prs)
        for a, b in prs:
            wr, lr = (a, b) if tier.MASS.iloc[a] > tier.MASS.iloc[b] else (b, a)
            win.append(wr); lose.append(lr); wt.append(w)
    return dict(win=np.array(win, int), lose=np.array(lose, int), wt=np.array(wt, float), tier=tier)


def fit_pairwise_logistic(dX, w, C, iters=100, tol=1e-9):
    """L2-regularized pairwise logistic (label always 1 for the win-lose row): minimize
    sum_i w_i * softplus(-dX_i . th) + (lam/2)||th||^2, lam = 1/C. Solved by damped Newton
    (the objective is convex, so damped Newton converges to the global optimum)."""
    n, d = dX.shape; lam = 1.0 / C; th = np.zeros(d)
    if n == 0: return th
    for _ in range(iters):
        z = dX @ th; p = _sigmoid(z)
        grad = dX.T @ (w * (p - 1.0)) + lam * th
        s = w * p * (1.0 - p)
        H = (dX * s[:, None]).T @ dX + lam * np.eye(d)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = grad
        th_new = th - step
        if np.max(np.abs(th_new - th)) < tol:
            th = th_new; break
        th = th_new
    return th


def fit_theta(cand_table, C=K.AUTO_C):
    """Return theta = (theta_smaller, theta_cutarea) from a per-candidate table (build_candidate_table)."""
    cand = add_tier_and_phi(cand_table); tier = cand[cand.in_tier]
    P = make_pairs(tier); T = P["tier"]
    phi = T[["phi_smaller", "phi_cutarea"]].values
    dX = phi[P["win"]] - phi[P["lose"]]
    theta = fit_pairwise_logistic(dX, P["wt"], C)
    return theta
