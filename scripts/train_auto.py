"""Fit CC-Auto's two weights theta from a TRAIN manifest (pairwise-logistic ranking; C=1.0).

python scripts/train_auto.py --op-dir <TRAIN_OPS_DIR> --manifest <TRAIN_MANIFEST.csv> --dataset microns \
       --out out/cc_auto_trained.json
Run on the corresponding MICrONS/FlyWire TRAIN split, this reproduces the shipped model weights.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contactcut import data, train_auto, constants as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", default=None); ap.add_argument("--C", type=float, default=K.AUTO_C)
    ap.add_argument("--out", default="cc_auto_trained.json"); a = ap.parse_args()
    man = data.read_manifest(a.manifest, a.dataset); ops = []
    for r in man.itertuples():
        p = data.op_npz_path(a.op_dir, r.operation_id)
        if not p.exists(): continue
        op = data.load_operation(p); op["operation_id"] = r.operation_id
        if a.dataset: op["dataset"] = a.dataset
        ops.append(op)
    table = train_auto.build_candidate_table(ops)
    theta = train_auto.fit_theta(table, C=a.C)
    obj = {"model": "CC-Auto (H_GLOBAL)", "trained_on": a.manifest, "selected_C": a.C,
           "phi": ["z(f_smaller)", "-z(log1p(f_cutarea))"], "theta": [float(theta[0]), float(theta[1])]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(obj, open(a.out, "w"), indent=2)
    print(f"[train_auto] {len(ops)} ops, {len(table)} candidates, C={a.C}  ->  theta = {theta}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
