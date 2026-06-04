"""Python reproduction of the SANDO planner (planner core + ROS I/O).

Mirrors the C++ structure under sando_ws/src/sando/. Designed so the Python and
C++ planners can be swapped at the node level by changing the executable in the
launch file. No NN distillation here — this is a faithful Python port for
modification.
"""

__version__ = "0.1.0"
