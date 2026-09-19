"""Native supervoxel contact graph from a merged object (GT-free).

Inputs:
  seg    : 3-D int array of native supervoxel ids on the local crop
  merged : 3-D bool mask of the pre-edit merged object (foreground)

leaf_graph returns the undirected contact graph:
  leaves : (n,) unique supervoxel ids present in the object
  Ei, Ej : (E,) endpoint node indices of each unique contact edge
  fc     : (E,) shared contact area = number of shared voxel faces (edge capacity c_e)
  cen    : (n, 3) supervoxel centroids (z, y, x voxel coords)
  vcount : (n,) native voxel count per supervoxel (its GT-free mass)
  lof    : (#fg voxels,) node index per foreground voxel
"""
import numpy as np


def leaf_graph(seg, merged):
    fg = merged
    leaves = np.unique(seg[fg]); leaves = leaves[leaves != 0].astype(np.uint64)
    n = len(leaves); lidx = {int(v): i for i, v in enumerate(leaves.tolist())}
    zz, yy, xx = np.nonzero(fg); lof = np.array([lidx[int(x)] for x in seg[fg]], np.int64)
    vcount = np.bincount(lof, minlength=n).astype(float)
    cz = np.bincount(lof, weights=zz, minlength=n) / vcount
    cy = np.bincount(lof, weights=yy, minlength=n) / vcount
    cx = np.bincount(lof, weights=xx, minlength=n) / vcount
    cen = np.stack([cz, cy, cx], 1)
    li3 = np.full(seg.shape, -1, np.int64); li3[fg] = lof
    pairs = {}
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3; s1[ax] = slice(None, -1); s2[ax] = slice(1, None)
        a_ = li3[tuple(s1)]; b_ = li3[tuple(s2)]; mk = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        if not mk.any(): continue
        aa = a_[mk]; bb = b_[mk]; lo_i = np.minimum(aa, bb); hi_i = np.maximum(aa, bb)
        keys = lo_i.astype(np.int64) * n + hi_i.astype(np.int64); uk, cnt = np.unique(keys, return_counts=True)
        for k, c in zip(uk.tolist(), cnt.tolist()): pairs[k] = pairs.get(k, 0) + c
    Ei = []; Ej = []; fc = []
    for k, c in pairs.items(): Ei.append(k // n); Ej.append(k % n); fc.append(c)
    return leaves, np.array(Ei, np.int64), np.array(Ej, np.int64), np.array(fc, float), cen, vcount, lof


def contact_centroids(seg, merged, leaves):
    """Centroid (z, y, x) of each contact interface, keyed by the ordered node-index pair (min, max)."""
    n = len(leaves); lm = {int(v): i for i, v in enumerate(leaves.tolist())}
    li3 = np.full(seg.shape, -1, np.int64); li3[merged] = np.array([lm[int(x)] for x in seg[merged]], np.int64)
    acc = {}
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3; s1[ax] = slice(None, -1); s2[ax] = slice(1, None)
        a_ = li3[tuple(s1)]; b_ = li3[tuple(s2)]; mk = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        if not mk.any(): continue
        zz, yy, xx = np.nonzero(mk); off = np.array([0., 0., 0.]); off[ax] = 0.5
        for i, j, zc, yc, xc in zip(a_[mk], b_[mk], zz, yy, xx):
            k = (min(i, j), max(i, j)); r = acc.get(k)
            if r is None: acc[k] = [zc + off[0], yc + off[1], xc + off[2], 1]
            else: r[0] += zc + off[0]; r[1] += yc + off[1]; r[2] += xc + off[2]; r[3] += 1
    return {k: np.array([r[0] / r[3], r[1] / r[3], r[2] / r[3]]) for k, r in acc.items()}
