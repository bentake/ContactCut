"""Fit the PairLogit within-panel ranker from a TRAIN manifest (sklearn logistic on 58-D feature differences).

python scripts/train_pairlogit.py --op-dir <TRAIN_OPS_DIR> --manifest <TRAIN_MANIFEST.csv> --dataset microns \
       --out out/pairlogit_trained.json
Requires scikit-learn. Run on the MICrONS CC-Search panels, this reproduces the shipped PairLogit weights.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contactcut import data, search, pairlogit, constants as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", default=None); ap.add_argument("--C", type=float, default=K.PAIRLOGIT_C)
    ap.add_argument("--out", default="pairlogit_trained.json"); a = ap.parse_args()
    man = data.read_manifest(a.manifest, a.dataset); panels = []
    for r in man.itertuples():
        p = data.op_npz_path(a.op_dir, r.operation_id)
        if not p.exists(): continue
        op = data.load_operation(p); op["operation_id"] = r.operation_id
        if a.dataset: op["dataset"] = a.dataset
        B = search.build_search(op)
        if len(B["cu"]) == 0: continue
        PH, QH = search.hfixed_base(B)
        if PH is None: continue
        rb = search.run_b45(B, PH, QH)
        panel = rb["panel"]
        if len(panel) < 2: continue
        X29 = pairlogit.panel_base29(panel, B)
        panels.append(dict(X29=X29, mass=np.array([mem["mass"] for mem in panel]),
                           hash=[mem["hash"] for mem in panel], oracle=rb["raw_oracle"]))
    model = pairlogit.train_pairlogit(panels, C=a.C)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(model, open(a.out, "w"), indent=2)
    print(f"[train_pairlogit] {len(panels)} panels  ->  coef dim {len(model['coef58'])}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
