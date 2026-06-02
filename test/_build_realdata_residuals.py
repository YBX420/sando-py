"""Build CA-prediction sup-norm residuals from REAL pedestrian trajectories
(ETH-UCY) for conformal recalibration of Q_alpha.

The planner predicts a human with a constant-acceleration (CA) model. The conformal
margin Q_alpha must cover sup_s ||c_true(s) - c_pred(s)|| -- the REAL deviation of a
pedestrian from a CA extrapolation. This script reproduces exactly that error on real
data and dumps the per-window sup-norm residuals so the hermetic test
stage4_conformal_realdata.py can split-conformal them WITHOUT a network dependency.

Pipeline (causal, mirrors the on-board EKF-CA predictor):
  - parse OpenTraj obsmat: cols [frame, id, x, _, y, vx, _, vy]; ground motion = (x, y)
  - group by pedestrian, sort by frame; dt auto-detected per dataset from vx ~ dx/dt
  - at each index i with W past points [i-W+1..i], LSQ-fit a per-axis quadratic (CA),
    extrapolate c_pred(t_i + k*dt) for k=1..H, compare to the TRUE future (i+k)
  - residual R_j = max_{k=1..H} ||c_true_k - c_pred_k||   (the over-horizon sup-norm)

Run (needs the three obsmat files cached in /tmp; downloads if absent):
  python3 test/_build_realdata_residuals.py
Writes test/data/eth_ucy_ca_residuals.npz
"""
import os
import sys
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

BASE = "https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets"
SETS = {
    "eth":   f"{BASE}/ETH/seq_eth/obsmat.txt",
    "hotel": f"{BASE}/ETH/seq_hotel/obsmat.txt",
    "zara01": f"{BASE}/UCY/zara01/obsmat.txt",
}
W = 4          # causal history points used for the CA (quadratic) fit
H = 3          # prediction horizon in steps (~1.2 s at dt=0.4)


def load(name, url):
    cache = os.path.join(DATA, f"obsmat_{name}.txt")
    if not os.path.exists(cache):
        urllib.request.urlretrieve(url, cache)
    raw = np.loadtxt(cache)
    frame, pid = raw[:, 0], raw[:, 1]
    x, y = raw[:, 2], raw[:, 4]
    vx, vy = raw[:, 5], raw[:, 7]
    return frame, pid, x, y, vx, vy


def detect_dt(frame, pid, x, vx):
    """dt per step from vx ~ dx/dframe-step: median over consecutive same-ped pairs."""
    dts = []
    for p in np.unique(pid):
        m = pid == p
        f = frame[m]; xx = x[m]; vv = vx[m]
        order = np.argsort(f)
        f, xx, vv = f[order], xx[order], vv[order]
        for i in range(len(f) - 1):
            dframe = f[i + 1] - f[i]
            if dframe <= 0:
                continue
            dx = xx[i + 1] - xx[i]
            v = 0.5 * (vv[i] + vv[i + 1])
            if abs(v) > 0.3:                       # only trust moving samples
                dts.append(dx / v)                 # = dt (seconds) for this step
    return float(np.median(dts)) if dts else 0.4


def ca_fit_predict(px, py, dt):
    """For one pedestrian's (px,py) sequence at uniform dt, produce sup-norm residuals
    of a causal CA (quadratic LSQ on W past pts) extrapolated H steps ahead."""
    n = len(px)
    out = []
    t = np.arange(W) * dt                          # local history times 0..(W-1)dt
    # design matrix for quadratic fit on the history window
    A = np.vstack([np.ones(W), t, t * t]).T        # (W,3)
    tf = (np.arange(W, W + H)) * dt                 # future times (continue the window)
    Af = np.vstack([np.ones(H), tf, tf * tf]).T     # (H,3)
    for i in range(W - 1, n - H):
        hx = px[i - W + 1:i + 1]; hy = py[i - W + 1:i + 1]
        cx, _, _, _ = np.linalg.lstsq(A, hx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, hy, rcond=None)
        predx = Af @ cx; predy = Af @ cy            # CA extrapolation, H steps
        truex = px[i + 1:i + 1 + H]; truey = py[i + 1:i + 1 + H]
        err = np.sqrt((truex - predx) ** 2 + (truey - predy) ** 2)   # (H,) ||e_k||
        out.append(float(err.max()))                # sup-norm over the horizon
    return out


all_res = []
per_scene = {}        # name -> residual array (for genuine cross-scene shift tests)
dt_used = 0.4
for name, url in SETS.items():
    try:
        frame, pid, x, y, vx, vy = load(name, url)
    except Exception as e:
        print(f"[skip] {name}: {e}")
        continue
    dt = detect_dt(frame, pid, x, vx)
    dt_used = dt
    res = []
    npeds = 0
    for p in np.unique(pid):
        m = pid == p
        f = frame[m]
        order = np.argsort(f)
        px = x[m][order]; py = y[m][order]
        if len(px) < W + H:
            continue
        npeds += 1
        res.extend(ca_fit_predict(px, py, dt))
    res = np.asarray(res, dtype=np.float64)
    per_scene[name] = res
    all_res.extend(res.tolist())
    print(f"{name:7s}: dt={dt:.3f}s  peds={npeds:4d}  windows={res.size:5d}  "
          f"median R={np.median(res):.3f}m  q90 R={np.quantile(res, 0.9):.3f}m")

all_res = np.asarray(all_res, dtype=np.float64)
print(f"\nTOTAL windows={all_res.size}  median={np.median(all_res):.3f}m  "
      f"q90={np.quantile(all_res, 0.9):.3f}m  q95={np.quantile(all_res, 0.95):.3f}m  max={all_res.max():.3f}m")

out = os.path.join(DATA, "eth_ucy_ca_residuals.npz")
# per-scene arrays stored under res_<scene> so the hermetic test can do a GENUINE
# cross-scene distribution-shift check (calibrate on one scene, deploy on another).
save_kw = {f"res_{k}": v for k, v in per_scene.items()}
np.savez(out, residuals=all_res, W=W, H=H, horizon_s=float(H * 0.4),
         datasets=list(per_scene.keys()), **save_kw)
print(f"saved {all_res.size} residuals (+ per-scene) -> {out}")
