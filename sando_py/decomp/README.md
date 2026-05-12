# decomp/ — convex decomposition (safety corridors)

Turns a polyline global path into a sequence of overlapping convex
polytopes (`Ax <= b`). Each corridor `i` covers segment `i` of the path
and overlaps corridors `i-1` and `i+1` so the local solver can smoothly
hand off between them.

The C++ project uses **DecompROS2** (ellipsoid + dilation). For the
Python port we start with **axis-aligned boxes** — simpler, still
correct, slightly more conservative. We can swap in something tighter
later if the local solver leaves obvious shortcuts on the table.

## Contract

```python
sfc(path:        List[np.ndarray],
    voxel_map:   VoxelMap,
    obstacles_t: List[DynTraj],
    params:      Parameters) -> List[Polytope]
```

## Files

| file       | status        | role                                                                    |
|------------|---------------|-------------------------------------------------------------------------|
| `aabb.py`  | implemented   | grow an axis-aligned box around a segment until it hits an obstacle or `sfc_size` |
| `sfc.py`   | implemented   | top-level: walks the path, calls `grow_aabb` per segment, returns `List[Polytope]` |

Covered by `tests/test_decomp.py`.

## V1 limitations

* `obstacles_t` is accepted but unused — the voxel map already carries
  inflated obstacle footprints from the HGP layer. A later variant can
  use it for per-obstacle ellipsoid dilation or time-aware corridors.
* `min_dist_from_agent_to_traj`, `inflate_unknown_boundary`,
  `dyn_base_inflation_m`, and `environment_assumption` are not yet
  consumed — only `sfc_size`, `drone_radius`, `z_min/z_max`,
  `use_shrinked_box`, and `shrinked_box_size` matter for v1.

## Notes

* `Polytope` is defined in `core.types` as `(A, b)` with `A: (n, 3)` and
  `b: (n, 1)` — same convention as the C++ struct.
* For an axis-aligned box centered at `c` with half-widths `(hx, hy, hz)`,
  `A` is the 6×3 stack of `±I_3` rows and `b[k] = c[k%3] ± h[k%3]`.

Parameters consumed (from `core.Parameters`):
  `sfc_size`, `min_dist_from_agent_to_traj`, `use_shrinked_box`,
  `shrinked_box_size`, `inflate_unknown_boundary`,
  `dyn_base_inflation_m`, `environment_assumption`.
