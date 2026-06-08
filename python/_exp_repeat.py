"""Quantify the NON-DETERMINISM: run the SAME config N times (sequentially, one process, so the wall-clock
solve budget sees normal load) and report the distribution of worst BODY clearance + how many runs COLLIDE.
If the same config straddles zero (some runs +, some -), the 'collision' is solve-truncation jitter around a
zero-margin path, not a deterministic trap. Reuses _audit_collisions.run.
Run:  python _exp_repeat.py <name> <N>     (name from _exp_intervene.EXP; default base_v6_stg, N=10)
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _audit_collisions as A
from _exp_intervene import EXP

name = sys.argv[1] if len(sys.argv) > 1 else "base_v6_stg"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = EXP[name]
print(f"=== {name} x{N}  cfg={cfg} ===")
worsts = []; ncoll = 0; nreach = 0
for i in range(N):
    r = A.run(**cfg)
    worsts.append(r["worst"]); ncoll += int(r["collided"]); nreach += int(r["reached"])
    print(f"  run {i+1:2d}: worstBody={r['worst']:+.3f} collided={int(r['collided'])} "
          f"reached={int(r['reached'])} final_x={r['final_x']:.1f} fails={r['nfail']}", flush=True)
w = np.array(worsts)
print(f"--- worstBody: min={w.min():+.3f} max={w.max():+.3f} mean={w.mean():+.3f} std={w.std():.3f} "
      f"| collided {ncoll}/{N}  reached {nreach}/{N} ---")
print(f"--- straddles zero (jitter, not trap): {bool(w.min() < 0 < w.max())} ---")
