"""Clean production-vs-stg matrix on the current build (inflation_hgp=yaml 0.45). stg OFF = production default."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _audit_collisions as A
for vm in (6, 9, 12):
    for stg in (False, True):
        r = A.run(vmax=vm, stg=stg, stdsd=0.25 if stg else None, t_max=30.0)
        flag = "  <-- COLLISION" if r["collided"] else ("" if r["reached"] else "  <-- no-reach")
        print("vmax=%2d stg=%d | worst=%+.3f collided=%d reached=%d fails=%d%s"
              % (vm, int(stg), r["worst"], r["collided"], r["reached"], r["nfail"], flag), flush=True)
