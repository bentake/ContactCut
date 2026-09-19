"""Self-contained native-3D two-way partition scorer (label-invariant).

The single entry point `score_common_3d(sl, gtl, node_weight, present)` scores one operation's
two-side supervoxel partition against the historical reference and returns a dict with, among others:
  MASS_3D_IOU  -> 3D Part. mIoU (native-voxel-mass-weighted mean side IoU, optimal side swap)
  MIN_SIDE_IOU -> MIN (smaller matched-side IoU)
  SUCCESS@10   -> S@10 (<=10% mislabeled native-supervoxel mass, nonempty minority side)
  VOI_MERGE, ARAND (ARI = 1 - ARAND)

Convention used throughout ContactCut (see pipeline.py): a leaf-level partition `sl` (len n) is
scored by expanding each leaf into its two GT sides, i.e. score(np.repeat(sl, 2), GT_ENT, W_ENT)
with GT_ENT = tile([0,1], n), W_ENT[0::2]=cA, W_ENT[1::2]=cB, so MASS is exact native-voxel mIoU.
"""
import numpy as np


# ---- MASS / SV partition IoU (optimal side swap) ---------------------------------------------------
def _wiou(pred, gt, m):
    inter = float(m[pred & gt].sum()); union = float(m[pred | gt].sum())
    return inter / union if union > 0 else 0.0


def _partition_iou(pred_side, gt_side, m, valid_pred):
    best = 0.0
    for flip in (False, True):
        p = np.where(valid_pred, (1 - pred_side) if flip else pred_side, -1)
        s = 0.0
        for side in (0, 1):
            s += _wiou(p == side, gt_side == side, m)
        best = max(best, 0.5 * s)
    return best


# ---- side IoU / precision-recall-F1 (optimal assignment) -------------------------------------------
def _iou(a, b, w):
    union = float(w[a | b].sum())
    return float(w[a & b].sum() / union) if union > 0 else float("nan")


def _best_assignment(pred_side, gt_side, weights, valid=None):
    pred = np.asarray(pred_side); gt = np.asarray(gt_side); w = np.asarray(weights, float)
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    best_mean, best = -1.0, (False, (0.0, 0.0))
    for swap in (False, True):
        assigned = np.where(valid, (1 - pred) if swap else pred, -1)
        s = [_iou(assigned == side, gt == side, w) for side in (0, 1)]
        sv = [v for v in s if np.isfinite(v)]
        m = float(np.mean(sv)) if sv else 0.0
        if m > best_mean:
            best_mean = m
            best = (swap, (0.0 if not np.isfinite(s[0]) else s[0], 0.0 if not np.isfinite(s[1]) else s[1]))
    return best


def _side_iou_metrics(pred_side, gt_side, w, valid=None):
    _, (i0, i1) = _best_assignment(pred_side, gt_side, w, valid)
    return {"SIDE_A_IOU": float(i0), "SIDE_B_IOU": float(i1), "MIN_SIDE_IOU": float(min(i0, i1))}


def _side_prf(pred_side, gt_side, w, valid=None):
    pred = np.asarray(pred_side); gt = np.asarray(gt_side); w = np.asarray(w, float)
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    swap, _ = _best_assignment(pred_side, gt_side, w, valid)
    assigned = np.where(valid, (1 - pred) if swap else pred, -1)
    ps, rs, fs = [], [], []
    for side in (0, 1):
        p_mask = assigned == side; g_mask = gt == side
        tp = float(w[p_mask & g_mask].sum()); pp = float(w[p_mask].sum()); gg = float(w[g_mask].sum())
        prec = tp / pp if pp > 0 else float("nan"); rec = tp / gg if gg > 0 else float("nan")
        f1 = (2 * tp / (pp + gg)) if (pp + gg) > 0 else float("nan")
        if np.isfinite(prec): ps.append(prec)
        if np.isfinite(rec): rs.append(rec)
        if np.isfinite(f1): fs.append(f1)
    return {"SIDE_PRECISION": float(np.mean(ps)) if ps else float("nan"),
            "SIDE_RECALL": float(np.mean(rs)) if rs else float("nan"),
            "SIDE_F1": float(np.mean(fs)) if fs else float("nan"),
            "MIN_SIDE_F1": float(np.min(fs)) if fs else float("nan")}


# ---- VOI / ARAND (mass-weighted contingency) -------------------------------------------------------
def _contingency(pred, gt, weights):
    pred = np.asarray(pred); gt = np.asarray(gt); w = np.asarray(weights, float)
    labels_p = np.unique(pred); labels_g = np.unique(gt)
    pi = {l: i for i, l in enumerate(labels_p)}; gi = {l: i for i, l in enumerate(labels_g)}
    C = np.zeros((len(labels_p), len(labels_g)), float)
    for a, b, ww in zip(pred, gt, w): C[pi[a], gi[b]] += ww
    return C


def _voi(pred_side, gt_side, weights):
    C = _contingency(pred_side, gt_side, weights); n = C.sum()
    if n <= 0: return {"VOI_SPLIT": 0.0, "VOI_MERGE": 0.0, "VOI_TOTAL": 0.0}
    P = C / n; pa = P.sum(axis=1, keepdims=True); pb = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.where(P > 0, np.log2(np.where(pb > 0, P / pb, 1.0)), 0.0)
        logs = np.where(P > 0, np.log2(np.where(pa > 0, P / pa, 1.0)), 0.0)
    h_pred_given_gt = -float((P * logm).sum()); h_gt_given_pred = -float((P * logs).sum())
    return {"VOI_SPLIT": h_pred_given_gt, "VOI_MERGE": h_gt_given_pred, "VOI_TOTAL": h_pred_given_gt + h_gt_given_pred}


def _adjusted_rand_error(pred_side, gt_side, weights):
    C = _contingency(pred_side, gt_side, weights); n = C.sum()
    if n <= 1: return {"ARAND": 0.0, "RAND_INDEX": 1.0}
    comb2 = lambda x: x * (x - 1.0) / 2.0
    sum_ij = comb2(C).sum(); a = C.sum(axis=1); b = C.sum(axis=0)
    sum_a = comb2(a).sum(); sum_b = comb2(b).sum(); total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0.0
    denom = 0.5 * (sum_a + sum_b) - expected
    ari = (sum_ij - expected) / denom if denom != 0 else 1.0
    return {"ARAND": float(1.0 - ari), "RAND_INDEX": float(ari)}


def score_common_3d(sl, gtl, node_weight, present=None, valid_pred=None):
    sl = np.asarray(sl); gtl = np.asarray(gtl); w = np.asarray(node_weight, float)
    U = np.ones(len(sl), bool) if present is None else np.asarray(present, bool)
    slU, gtU, wU = sl[U], gtl[U], w[U]
    vpU = np.ones(int(U.sum()), bool) if valid_pred is None else np.asarray(valid_pred, bool)[U]
    out = {}
    out["MASS_3D_IOU"] = float(_partition_iou(slU, gtU, wU, vpU))
    out["SV_3D_IOU"] = float(_partition_iou(slU, gtU, np.ones_like(wU), vpU))
    si = _side_iou_metrics(slU, gtU, wU); pf = _side_prf(slU, gtU, wU)
    out["MIN_SIDE_IOU"] = float(si["MIN_SIDE_IOU"])
    for k in ("SIDE_PRECISION", "SIDE_RECALL", "SIDE_F1", "MIN_SIDE_F1"): out[k] = float(pf[k])
    out["IOU80"] = float(out["MASS_3D_IOU"] >= 0.80); out["IOU90"] = float(out["MASS_3D_IOU"] >= 0.90)
    out["MINSIDE_GE_050"] = float(out["MIN_SIDE_IOU"] >= 0.50)
    out["MINSIDE_GE_080"] = float(out["MIN_SIDE_IOU"] >= 0.80)
    out["MINSIDE_GE_090"] = float(out["MIN_SIDE_IOU"] >= 0.90)
    tot = max(float(wU.sum()), 1.0)
    ma = float(wU[slU == 0].sum()) / tot; minor = min(ma, 1 - ma)
    same = float(wU[slU == gtU].sum()); mass = 1.0 - max(same, tot - same) / tot
    for k in (5, 10, 15): out[f"SUCCESS@{k}"] = float(minor >= 0.01 and mass <= k / 100.0)
    out.update({kk: float(vv) for kk, vv in _voi(slU, gtU, wU).items()})
    out.update({kk: float(vv) for kk, vv in _adjusted_rand_error(slU, gtU, wU).items()})
    return out
