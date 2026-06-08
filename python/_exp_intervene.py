"""Intervention matrix for the FORCED-vs-MISSED question: run the same colliding scene (vmax=6) under
single-knob changes and report worst BODY clearance + reached. If ANY reasonable planner config CLEARS the
static wall (worst>=0) AND still reaches, then a collision-free path existed -> the base collision was a
MISSED decision, not a forced trap. Reuses _audit_collisions.run (densified body-clearance, all boxes).
Run:  python _exp_intervene.py <name>      names: see EXP below; 'all' runs every one.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _audit_collisions as A

EXP = {
    # reproduce the collision (ST-graph on, the gif config)
    "base_v6_stg":   dict(vmax=6.0, stg=True,  stdsd=0.25, t_max=30.0),
    # production at vmax=6 (ST-graph off) -- does it hold honestly or also collide?
    "v6_stg_off":    dict(vmax=6.0, stg=False, t_max=30.0),
    # space-time corridor ON -> keeps w_corridor active (planner.hpp:1063) + carves static-free cuboids,
    # movers still hard. If this clears the static wall the path was feasible with movers still handled.
    "v6_stc_on":     dict(vmax=6.0, stc=True,  stdsd=0.25, t_max=30.0),
    # classic SFC corridor ON by disabling dynamic-mode (dyn=0 stops planner.hpp:1063 zeroing w_corridor);
    # caveat: movers become soft in this mode.
    "v6_corridor_on": dict(vmax=6.0, stg=True, stdsd=0.25, dyn=0.0, wcor=100.0, t_max=30.0),
    # force ALL obstacles hard (tests whether the hard certificate would keep the body off the static wall;
    # AABB hard path may be a no-op -> if so this won't clear, which is itself informative).
    "v6_force_hard": dict(vmax=6.0, avoid="hard", t_max=30.0),
    # the production headline point for contrast (razor-thin but 'safe')
    "v12_prod":      dict(vmax=12.0, t_max=30.0),
}


def one(name):
    r = A.run(**EXP[name])
    verdict = ("CLEARED (path existed)" if (r["reached"] and not r["collided"] and r["worst"] >= 0.0)
               else "COLLIDED" if r["collided"] else "HOLD/no-reach")
    print(f"{name:16s} worstBody={r['worst']:+.3f} reached={int(r['reached'])} collided={int(r['collided'])} "
          f"final_x={r['final_x']:.1f} fails={r['nfail']}  -> {verdict}")
    return r


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(EXP) if sel == "all" else [sel]
    for n in names:
        one(n)
