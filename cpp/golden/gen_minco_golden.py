"""Golden-data generator for MinjerkTraj (minco.py) C++ port verification.

不许欺骗、必须还原:这里用真 Python MinjerkTraj 跑出「输入 -> 输出」金标准,
C++ 移植后读同样输入、算出来必须逐项对上(容差内),对不上就不算还原。

dump 到 cpp/golden/minco_cases.txt(纯文本,C++ 无需 JSON 依赖即可解析)。
跑: python cpp/golden/gen_minco_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.minco import MinjerkTraj

rng = np.random.default_rng(12345)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []
cases = []
# 覆盖 M=1..6,随机途经点 / 时长 / 起末边界
for M in [1, 2, 3, 4, 5, 6]:
    wp = rng.uniform(-5.0, 5.0, (M + 1, 3))
    T = rng.uniform(0.3, 1.6, M)
    v0 = rng.normal(0, 0.5, 3); a0 = rng.normal(0, 0.3, 3)
    vf = rng.normal(0, 0.5, 3); af = rng.normal(0, 0.3, 3)
    mj = MinjerkTraj(wp, T, v0=v0, a0=a0, vf=vf, af=af)

    K = 50
    ts = np.linspace(0.0, mj.t_end, K)
    e0 = mj.eval_deriv(ts, 0)   # (K,3)
    e1 = mj.eval_deriv(ts, 1)
    e2 = mj.eval_deriv(ts, 2)
    e3 = mj.eval_deriv(ts, 3)
    cp = mj.control_points()        # (M,6,3)
    cpt = mj.control_point_times()  # (M,6)
    # M1 analytic gradient
    J, gq, gT = mj.energy_grad()
    dcc = rng.normal(0, 1.0, (6 * M, 3))
    dct = rng.normal(0, 1.0, M)
    gq2, gT2 = mj.grad_from_dcost_dc(dcc, dcost_dT_explicit=dct)
    cpdt = mj.control_points_dT_explicit()  # (M,6,3)

    lines.append("CASE")
    lines.append(f"M {M}")
    lines.append(f"T {fmt(T)}")
    lines.append(f"WP {fmt(wp)}")
    lines.append(f"V0 {fmt(v0)}")
    lines.append(f"A0 {fmt(a0)}")
    lines.append(f"VF {fmt(vf)}")
    lines.append(f"AF {fmt(af)}")
    lines.append(f"TEND {repr(float(mj.t_end))}")
    lines.append(f"C {fmt(mj.c)}")            # (6M,3) row-major
    lines.append(f"K {K}")
    lines.append(f"TS {fmt(ts)}")
    lines.append(f"E0 {fmt(e0)}")
    lines.append(f"E1 {fmt(e1)}")
    lines.append(f"E2 {fmt(e2)}")
    lines.append(f"E3 {fmt(e3)}")
    lines.append(f"CP {fmt(cp)}")             # (M,6,3)
    lines.append(f"CPT {fmt(cpt)}")           # (M,6)
    lines.append(f"ENERGY {repr(float(J))}")
    lines.append(f"EGQ {fmt(gq)}")            # (M-1,3)
    lines.append(f"EGT {fmt(gT)}")            # (M)
    lines.append(f"DCC {fmt(dcc)}")           # (6M,3) input
    lines.append(f"DCT {fmt(dct)}")           # (M) input
    lines.append(f"GQ2 {fmt(gq2)}")           # grad_from_dcost_dc -> (M-1,3)
    lines.append(f"GT2 {fmt(gT2)}")           # (M)
    lines.append(f"CPDT {fmt(cpdt)}")         # (M,6,3)
    lines.append("END")

out = os.path.join(os.path.dirname(__file__), "minco_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l=='CASE'])} cases)")
