"""ContactCut: native-3D intervention selection for false-merge correction.

Public API:
  from contactcut import data, pipeline, search, pairlogit, train_auto, constants
  op = data.load_operation("op.npz")
  rows = pipeline.run_operation(op, theta)                 # CC-Auto + baselines (Table 1 / 2a)
  B = search.build_search(op); PH, QH = search.hfixed_base(B)
  rb = search.run_b45(B, PH, QH)                           # CC-Search reachability (Table 2b)
"""
from . import constants, data, graph, scoring, solvers, pipeline, search, pairlogit, train_auto

__all__ = ["constants", "data", "graph", "scoring", "solvers", "pipeline", "search", "pairlogit", "train_auto"]
__version__ = "1.0.0"
