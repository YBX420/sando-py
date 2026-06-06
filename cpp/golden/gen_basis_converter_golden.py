import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.types import BasisConverter
def fmt(a): return " ".join(repr(float(x)) for x in np.asarray(a,float).reshape(-1))
bc = BasisConverter()
lines=[]
for num in [2,3,4,5,7]:
    mvp=bc.get_minvo_pos_converters(num); bep=bc.get_bezier_pos_converters(num)
    mvv=bc.get_minvo_vel_converters(num); mva=bc.get_minvo_accel_converters(num)
    absp=bc.get_A_bspline(num)
    lines+=["CASE",f"NUM {num}",
            f"MVP {fmt(np.array(mvp))}", f"BEP {fmt(np.array(bep))}",
            f"MVV {fmt(np.array(mvv))}", f"MVA {fmt(np.array(mva))}",
            f"ABS {fmt(np.array(absp))}","END"]
out=os.path.join(os.path.dirname(__file__),"basis_converter_cases.txt")
open(out,"w").write("\n".join(lines)+"\n"); print("wrote",out)
