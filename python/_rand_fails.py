"""Run N random scenes; render a gif ONLY for the problematic ones (COLLIDED or ON-HOLD/not-reached).
Phase 1: classify all N by closed-loop rollout (no render). Phase 2: render only the failures/holds,
named media/_rand_COLL_NN.gif (collision) or _rand_HOLD_NN.gif (held, no reach). Clean reaches get no gif.
Run:  python _rand_fails.py [N=100] [workers=6]
"""
import os, sys
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX"):
    os.environ.pop(k, None)
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _rand_scenes as R


def _classify(seed):
    fr, S, G, reached, t, worst = R.record(seed)
    fx = float(fr[-1][1][0]) if fr else 0.0
    return (seed, bool(reached), bool(worst < 0.0), float(worst), float(t), fx)


def _render(seed):
    try:
        R.render_one(seed)                       # writes media/_rand_SEED.gif
        return (seed, True, "")
    except Exception as e:
        return (seed, False, f"{type(e).__name__}: {e}")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    print(f"=== phase 1: classify {N} scenes ===", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_classify, range(N)):
            rows.append(r)
            seed, reached, coll, worst, t, fx = r
            tag = "COLL" if coll else ("HOLD" if not reached else "ok")
            if tag != "ok":
                print(f"  scene {seed:02d}: {tag:4s} reached={int(reached)} worst={worst:+.3f} final_x={fx:4.1f} t={t:.1f}", flush=True)
    interesting = [r for r in rows if r[2] or not r[1]]      # collided OR not reached
    nreach = sum(r[1] for r in rows); ncoll = sum(r[2] for r in rows)
    nhold = sum((not r[1]) and (not r[2]) for r in rows)
    print(f"=== {N}: reached {nreach}, collided {ncoll}, held {nhold}. rendering {len(interesting)} gifs ===", flush=True)
    seeds = [r[0] for r in interesting]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = {s: (okc, err) for s, okc, err in ex.map(_render, seeds)}
    # rename rendered gifs with COLL/HOLD tag
    cls = {r[0]: ("COLL" if r[2] else "HOLD") for r in interesting}
    for s in seeds:
        if not done.get(s, (False,))[0]:
            print(f"  render scene {s:02d} FAILED: {done[s][1]}"); continue
        src = os.path.join(ROOT, "media", f"_rand_{s:02d}.gif")
        dst = os.path.join(ROOT, "media", f"_rand_{cls[s]}_{s:02d}.gif")
        if os.path.exists(dst): os.remove(dst)
        if os.path.exists(src): os.rename(src, dst)
    print("=== done. problematic gifs: ===")
    for r in sorted(interesting, key=lambda x: (not x[2], x[0])):
        seed, reached, coll, worst, t, fx = r
        print(f"  media/_rand_{cls[seed]}_{seed:02d}.gif  worst={worst:+.3f} final_x={fx:4.1f}")


if __name__ == "__main__":
    main()
