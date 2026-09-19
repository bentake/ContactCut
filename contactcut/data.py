"""Operation I/O for ContactCut.

Each operation is one localized false-merge correction, stored as a single .npz with this schema
(the "raw" per-operation inputs the pipeline consumes; produced from CAVE/FlyWire — see README):

  seg      : (Z, Y, X) int   native supervoxel id volume on the local crop
  merged   : (Z, Y, X) bool  pre-edit merged object mask (foreground)
  lo       : (3,) int        crop origin (z, y, x) in native voxels
  sv       : (Z, Y, X) int   CAVE supervoxel id per voxel (for GT lookup); aligned with seg
  gtl      : (Ngt,) int      GT side (0/1) per GT node id (the two post-edit descendants)
  nid      : (Ngt,) int      GT node ids indexing gtl
  N        : int             number of GT nodes (len(gtl))
  interface_xyz_nm : (3,)    recorded proofreading interface cue x0 in nm (x, y, z / CAVE order)
  dataset  : str             'microns' or 'flywire'

A manifest is a CSV with at least column `operation_id`; `dataset` may be a column or a CLI default.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_operation(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return dict(
        seg=z["seg"], merged=z["merged"].astype(bool), lo=z["lo"],
        sv=z["sv"], gtl=z["gtl"].astype(int), nid=z["nid"], N=int(z["N"]),
        interface_xyz_nm=np.asarray(z["interface_xyz_nm"], float),
        dataset=str(z["dataset"]) if "dataset" in z else "microns",
    )


def read_manifest(manifest_csv, default_dataset=None):
    df = pd.read_csv(manifest_csv); df["operation_id"] = df["operation_id"].astype(str)
    if "dataset" not in df.columns:
        if default_dataset is None:
            raise ValueError("manifest has no 'dataset' column; pass --dataset")
        df["dataset"] = default_dataset
    return df


def op_npz_path(op_dir, operation_id):
    return Path(op_dir) / f"{operation_id}.npz"
