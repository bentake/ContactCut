"""Run CC-Search bounded-search reachability and PairLogit panel ordering (NEAR05@k).

python scripts/run_search.py --op-dir <OPS_DIR> --manifest <MANIFEST.csv> --dataset microns \
       --pairlogit models/pairlogit_microns.json --out out/search
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contactcut import data, search, pairlogit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", default=None); ap.add_argument("--pairlogit", default=None)
    ap.add_argument("--out", default="out"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model = json.load(open(a.pairlogit)) if a.pairlogit else None
    man = data.read_manifest(a.manifest, a.dataset)
    rows = []; near = {"FACILITY": {k: [] for k in (1, 2, 3, 5)}, "PAIRLOGIT": {k: [] for k in (1, 2, 3, 5)}}
    for r in man.itertuples():
        p = data.op_npz_path(a.op_dir, r.operation_id)
        if not p.exists(): continue
        op = data.load_operation(p); op["operation_id"] = r.operation_id
        if a.dataset: op["dataset"] = a.dataset
        B = search.build_search(op)
        if len(B["cu"]) == 0: continue
        PH, QH = search.hfixed_base(B)
        if PH is None: continue
        best_single = search.best_single_oracle(B)
        rb = search.run_b45(B, PH, QH)
        rows.append(dict(operation_id=r.operation_id, dataset=op["dataset"], H_FIXED=rb["base_mass"],
                         Best_single=best_single, B45_pair=rb["raw_oracle"], Best_in_K5=rb["k5_disp"],
                         n_contacts=int(rb["nC"]), n_basins=int(rb["n_unique"])))
        # NEAR05@k for facility order vs PairLogit order (base + first k alternatives within 0.05 of pool oracle)
        panel = rb["panel"]; oracle = rb["raw_oracle"]; thr = oracle - 0.05
        masses = [mem["mass"] for mem in panel]  # panel[0] is base
        fac_alts = list(range(1, len(panel)))    # facility greedy order == panel order (search.run_b45)
        orders = {"FACILITY": fac_alts}
        if model is not None:
            orders["PAIRLOGIT"] = pairlogit.order_panel(panel, B, model)
        for name, order in orders.items():
            for k in (1, 2, 3, 5):
                topk = [0] + order[:k]                      # base + first k ranked alternatives
                near[name][k].append(int(max(masses[i] for i in topk) >= thr))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "search_per_op.csv"), index=False)
    reach = df[["H_FIXED", "Best_single", "B45_pair", "Best_in_K5"]].mean()
    print(f"[run_search] {len(df)} contact-valid operations, dataset={a.dataset or 'per-manifest'}")
    print("Reachability (mean 3D Part. mIoU):")
    print(reach.to_string())
    summ = [{"metric": kk, "mean": float(vv)} for kk, vv in reach.items()]
    if any(near["FACILITY"][k] for k in (1, 2, 3, 5)):
        print("\nNEAR05@k (fraction reaching within 0.05 of the base+B45 pool oracle):")
        for name in orders:
            line = {k: round(float(np.mean(near[name][k])), 4) for k in (1, 2, 3, 5)}
            print(f"  {name:9s}", line)
            summ.append({"metric": f"NEAR05_{name}", **{f"@{k}": line[k] for k in (1, 2, 3, 5)}})
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "search_summary.csv"), index=False)
    print(f"\nwrote {a.out}/search_per_op.csv and search_summary.csv")


if __name__ == "__main__":
    main()
