"""Distribution stats over N scenes (worst body clearance, flight time, obstacle count) to confirm the
scenes are genuinely challenging, not trivially open. Set HARD=1 for the complex set.
Run:  python _rand_stats.py [N=100] [workers=6]
"""
import os, sys
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX", "RETIME"):
    os.environ.pop(k, None)
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _rand_scenes as R


def one(seed):
    import numpy as np
    fr, S, G, reached, t, worst = R.record(seed)
    nobs = len(R.make_scene(seed))
    pos = np.array([f[1] for f in fr]) if len(fr) > 1 else np.zeros((1, 3))
    arc = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
    straight = float(np.linalg.norm(np.asarray(G) - np.asarray(S)))
    detour = arc / max(straight, 1e-6)
    meanlat = float(np.mean(np.abs(pos[:, 1]))); maxlat = float(np.max(np.abs(pos[:, 1])))
    avg = arc / max(t, 1e-6)
    return (seed, int(reached), int(worst < 0), float(worst), float(t), nobs, float(avg), detour, meanlat, maxlat)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, range(N)))
    w = np.array([r[3] for r in rows]); t = np.array([r[4] for r in rows]); nob = np.array([r[5] for r in rows])
    sp = np.array([r[6] for r in rows])
    reach = sum(r[1] for r in rows); coll = sum(r[2] for r in rows)
    print(f"HARD={os.environ.get('HARD','0')} GOAL_X={R.GOAL_X} vmax={R.VMAX} N={N}: reached {reach}, collided {coll}")
    print(f"  AVG SPEED      : median={np.median(sp):.2f} m/s  p10={np.percentile(sp,10):.2f}  max={sp.max():.2f}  (vmax={R.VMAX})")
    det = np.array([r[7] for r in rows]); ml = np.array([r[8] for r in rows]); xl = np.array([r[9] for r in rows])
    print(f"  PATH detour    : median={np.median(det):.3f}x (arc/straight)  p90={np.percentile(det,90):.3f}  max={det.max():.3f}")
    print(f"  lateral |y|    : mean(median)={np.median(ml):.2f} m  max(median)={np.median(xl):.2f} m  max(p90)={np.percentile(xl,90):.2f}")
    print(f"  obstacles/scene: min={nob.min()} median={int(np.median(nob))} max={nob.max()}")
    print(f"  worst body clr : min={w.min():+.3f} p10={np.percentile(w,10):+.3f} median={np.median(w):+.3f}")
    print(f"  flight time    : median={np.median(t):.1f}s p90={np.percentile(t,90):.1f}s max={t.max():.1f}s")
    print(f"  tight passes (worst<0.30): {int((w<0.30).sum())}/{N}   very tight (worst<0.15): {int((w<0.15).sum())}/{N}")
    # show the 6 tightest
    order = np.argsort(w)[:6]
    print("  tightest scenes:", "  ".join(f"#{rows[i][0]}={w[i]:+.2f}(t{t[i]:.0f})" for i in order))


if __name__ == "__main__":
    main()
