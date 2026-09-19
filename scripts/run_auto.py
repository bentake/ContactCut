"""Run CC-Auto automatic correction and the selector x solver ablation.

python scripts/run_auto.py --op-dir <OPS_DIR> --manifest <MANIFEST.csv> --dataset microns \
       --model models/cc_auto_microns.json --out out/microns
Add --transfer to score the FlyWire manifest with the MICrONS-trained theta (CC-Auto_{M->FW}).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contactcut import data, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", default=None, help="microns|flywire (default column in manifest)")
    ap.add_argument("--model", required=True, help="cc_auto_*.json with the learned theta")
    ap.add_argument("--out", default="out"); ap.add_argument("--transfer", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    theta = np.asarray(json.load(open(a.model))["theta"], float)
    man = data.read_manifest(a.manifest, a.dataset)
    rows = []
    for r in man.itertuples():
        p = data.op_npz_path(a.op_dir, r.operation_id)
        if not p.exists(): continue
        op = data.load_operation(p); op["operation_id"] = r.operation_id
        if a.dataset: op["dataset"] = a.dataset
        res = pipeline.run_operation(op, theta)
        for method, sc in res.items():
            if not sc: continue
            rows.append({"operation_id": r.operation_id, "dataset": op["dataset"], "method": method, **sc})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "auto_per_op.csv"), index=False)
    # FA-3D Part. mIoU: invalid (missing) outputs score 0 over the eligible cohort; here all methods emit, so
    # FA == MASS on the common cohort. Coverage = fraction of operations a method produced a valid partition for.
    n_ops = df.operation_id.nunique()
    agg = df.groupby("method").agg(N=("operation_id", "nunique"), MASS=("MASS", "mean"), MIN=("MIN", "mean"),
                                   S10=("S10", "mean"), ARI=("ARI", "mean"), VOI=("VOI", "mean")).reset_index()
    agg["Coverage"] = agg["N"] / n_ops
    agg = agg.sort_values("MASS", ascending=False)
    agg.to_csv(os.path.join(a.out, "auto_summary.csv"), index=False)
    print(f"[run_auto] {n_ops} operations, dataset={a.dataset or 'per-manifest'}")
    print(agg.to_string(index=False))
    print(f"\nwrote {a.out}/auto_per_op.csv and auto_summary.csv")


if __name__ == "__main__":
    main()
