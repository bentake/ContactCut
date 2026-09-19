"""Fixed configuration constants for ContactCut.

Voxel resolutions, candidate radius, min-cut terminal capacity, and the CC-Auto / H_FIXED / CC-Search
hyperparameters. Rationale for each non-obvious value is noted inline.
"""
import numpy as np

# --- voxel resolution (z, y, x), nanometres ---------------------------------------------------------
RES0_MICRONS = np.array([40.0, 8.0, 8.0])     # MICrONS minnie65 native voxel size (z, y, x)
RES0_FLYWIRE = np.array([40.0, 16.0, 16.0])   # FlyWire FAFB native voxel size (z, y, x)
CVRES = np.array([8.0, 8.0, 40.0])            # CAVE coordinate order (x, y, z) -> nm scaling for the interface cue

# --- candidate enumeration --------------------------------------------------------------------------
RADIUS_NM = 500.0        # candidate contacts within 500 nm of the interface cue x0
                         # (smallest radius with full candidate coverage in the radius-sensitivity sweep)

# --- seeded s-t min cut -----------------------------------------------------------------------------
# Terminal-arc capacity: INF = 8 * (sum_i weighted_degree_i + 1) = 16 * sum_e c_e + 8.
# Exceeds the most expensive all-native cut, so pinned endpoints are never separated by the min cut.
# Any multiplier keeping INF > sum_e c_e suffices; 8 is a conservative safety margin.
INF_MULTIPLIER = 8

# --- CC-Auto (H_GLOBAL) -----------------------------------------------------------------------------
Z_CLIP_AUTO = 8.0        # within-tier robust-z clip for the two Auto features
PAIR_MARGIN = 0.02       # winner/loser pair requires |dMASS| >= 0.02 (near-tie pairs discarded)
PAIR_CAP = 4096          # max training pairs per operation (deterministic md5 subsample)
PAIR_SEED_BASE = 20260901
AUTO_C = 1.0             # L2 pairwise-logistic inverse-regularization (selected by component-grouped CV)

# --- H_FIXED scalar (untuned lexicographic-on-connectivity, then a fixed smaller/cutarea trade-off) --
HFIX_W_BC = 1e9          # both-sides-connected  (connectivity priority)
HFIX_W_NEXTRA = 1e5      # extra connected components (connectivity priority)
HFIX_W_SMALLER = 1e2     # smaller-side mass fraction (within-tier trade-off term)
HFIX_W_CUTAREA = 1e-3    # severed contact area       (within-tier trade-off term)

# --- CC-Search (B45 + Facility-K5 + PairLogit) ------------------------------------------------------
B_PAIR = 45              # paired-contact budget (saturation point of the budget sweep)
B45_SEED = 20260827      # deterministic pair-sampling seed
K_ALT = 5               # facility-location panel size (base H_FIXED + K alternatives)
Z_CLIP_PAIRLOGIT = 5.0   # within-panel robust-z clip for the PairLogit features
GOOD05_MARGIN = 0.05     # PairLogit positive label: MASS within 0.05 of the base+B45 pool oracle
PAIRLOGIT_C = 1.0

def res0_for(dataset: str) -> np.ndarray:
    d = dataset.lower()
    if d.startswith("micron"): return RES0_MICRONS
    if d.startswith("fly"):    return RES0_FLYWIRE
    raise ValueError(f"unknown dataset {dataset!r} (expected 'microns' or 'flywire')")
