# core/

Shared types and pure-math utilities. No I/O, no networking, no algorithms
above the level of single-function transforms — anything heavier lives in a
sibling subpackage.

## Files

| file                | C++ origin                | what it provides                                        |
|---------------------|---------------------------|---------------------------------------------------------|
| `types.py`          | `sando_type.hpp:20-205`   | `StateDeriv`, `Polytope`, `RobotState`, `Parameters`    |
| `piecewise_poly.py` | `sando_type.hpp:545-720`  | `PieceWisePol` (eval / velocity / acceleration)         |
| `dyn_traj.py`       | `sando_type.hpp:724-985`  | `DynTraj` (Analytic via sympy, Piecewise via PieceWisePol) |
| `basis_converter.py`| `sando_type.hpp:210-538`  | MINVO / Bezier / BSpline basis matrices and converters  |
| `utils.py`          | `utils.hpp` + `utils.cpp` | `angle_wrap`, `getMinTimeDoubleIntegrator{1,3}D`, `projectPointToBox`, color id ints, ... |
| `config_loader.py`  | `sando.yaml` + ROS params | `load_parameters_from_yaml()` → `Parameters` + extras   |

## Status

Implemented. Covered by `tests/test_core_types.py`, `tests/test_utils.py`,
`tests/test_config_loader.py`.

## Usage

```python
from sando_py.core import Parameters, RobotState, DynTraj, TrajMode, PieceWisePol
from sando_py.core import load_parameters_from_yaml
from sando_py.core import utils  # angle_wrap, getMinTime..., ...
```

## Notes

* Field names, defaults, and types in `Parameters` mirror the C++ struct
  one-for-one (with line-number cross-references in comments) so YAML
  configs stay shareable between the two implementations.
* `DynTraj` uses `sympy.lambdify` in place of C++ `exprtk`. Same intrinsics
  (`sin`, `cos`, `exp`, `log`, ...).
* `BasisConverter` constants are copied verbatim from the C++ source so
  results match to within IEEE-754 rounding of the matrix inverse.
