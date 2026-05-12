"""sando_py — Python re-implementation of the MIT-ACL SANDO planner that
drops into the upstream ROS 2 simulation world.

The C++ reference implementation lives in ~/code/sando_ws/src/sando. See
sando_py/README.md for the layout and the per-tick flow.

Subpackages
-----------
core/   shared dataclasses + pure-math utilities (no I/O, no algorithms)
hgp/    global planner — heat-map A* over a voxel occupancy grid
decomp/ convex decomposition — polyline path -> safety corridors (Ax <= b)
solver/ local trajectory optimizer — smooth piecewise polynomial in the corridors
sim_io/ ROS adapter layer — rclpy.Node + message converters that wrap the
        algorithm code so the upstream RViz / dynamic_forest / fake_sim /
        goal_sender world can stay unchanged. Only this subpackage imports
        rclpy; the other four are pure Python and importable without ROS.
"""

__version__ = "0.0.1"
