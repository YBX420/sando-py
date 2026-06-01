"""Demo obstacle publisher for the per-class RViz demo.

Publishes two obstacles on `predicted_trajs` (dynus_interfaces/DynTraj) so the
planner sees them, plus its OWN visualization MarkerArray on `perclass_obstacles`
(sando_node draws no obstacle markers):

  - HUMAN  (id 100, HARD): a person crossing the drone's path, analytic motion.
  - WALL   (id 200, SOFT): a static thin box the drone may graze.

The id range is the swappable class tag the planner reads
(200<=id<300 -> wall/SOFT, else human/HARD).
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from dynus_interfaces.msg import DynTraj


class PerClassObstaclePub(Node):
    def __init__(self):
        super().__init__("perclass_obstacle_pub")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("human_x", 5.0)
        self.declare_parameter("human_vy", 0.8)
        self.declare_parameter("human_y0", -3.0)
        self.declare_parameter("human_z", 1.0)
        self.declare_parameter("wall_center", [5.0, 1.2, 2.0])
        self.declare_parameter("wall_extents", [2.0, 0.4, 3.0])
        self.frame = self.get_parameter("frame_id").value
        self.hx = float(self.get_parameter("human_x").value)
        self.vy = float(self.get_parameter("human_vy").value)
        self.hy0 = float(self.get_parameter("human_y0").value)
        self.hz = float(self.get_parameter("human_z").value)
        self.wc = [float(v) for v in self.get_parameter("wall_center").value]
        self.we = [float(v) for v in self.get_parameter("wall_extents").value]
        self.human_bbox = [0.6, 0.6, 1.8]

        self.t0 = self._now()
        self.pub_traj = self.create_publisher(DynTraj, "predicted_trajs", 10)
        self.pub_viz = self.create_publisher(MarkerArray, "perclass_obstacles", 10)
        self.create_timer(0.1, self._tick)   # 10 Hz

    def _now(self) -> float:
        t = self.get_clock().now().to_msg()
        return float(t.sec) + 1e-9 * float(t.nanosec)

    def _human_pos(self, t: float):
        return [self.hx, self.hy0 + self.vy * (t - self.t0), self.hz]

    def _dyntraj(self, tid, bbox, fx, fy, fz, vx, vy, vz, pos):
        m = DynTraj()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.id = int(tid)
        m.is_agent = False
        m.bbox = [float(b) for b in bbox]
        m.mode = "analytic"
        m.function = [fx, fy, fz]
        m.velocity = [vx, vy, vz]
        m.pos = Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.goal = [float(pos[0]), float(pos[1]), float(pos[2])]
        return m

    def _tick(self):
        now = self._now()
        hp = self._human_pos(now)
        # HUMAN (HARD, id 100): analytic linear crossing in y, t is absolute ROS sec
        human = self._dyntraj(
            100, self.human_bbox,
            f"{self.hx}", f"{self.hy0} + {self.vy}*(t - {self.t0})", f"{self.hz}",
            "0.0", f"{self.vy}", "0.0", hp)
        # WALL (SOFT, id 200): static box
        wall = self._dyntraj(
            200, self.we,
            f"{self.wc[0]}", f"{self.wc[1]}", f"{self.wc[2]}",
            "0.0", "0.0", "0.0", self.wc)
        self.pub_traj.publish(human)
        self.pub_traj.publish(wall)
        self._publish_markers(hp)

    def _publish_markers(self, hp):
        arr = MarkerArray()
        # human body (red sphere)
        body = Marker()
        body.header.frame_id = self.frame
        body.header.stamp = self.get_clock().now().to_msg()
        body.ns = "human"; body.id = 0; body.type = Marker.SPHERE; body.action = Marker.ADD
        body.pose.position = Point(x=hp[0], y=hp[1], z=hp[2])
        body.pose.orientation.w = 1.0
        r = 2.0 * 0.5 * max(self.human_bbox)
        body.scale = Vector3(x=r, y=r, z=r)
        body.color = ColorRGBA(r=0.85, g=0.1, b=0.1, a=1.0)
        arr.markers.append(body)
        # human d_safe shell (translucent red), d_safe(human)=0.8
        ring = Marker()
        ring.header = body.header
        ring.ns = "human_dsafe"; ring.id = 1; ring.type = Marker.SPHERE; ring.action = Marker.ADD
        ring.pose = body.pose
        dr = 2.0 * (0.5 * max(self.human_bbox) + 0.8)
        ring.scale = Vector3(x=dr, y=dr, z=dr)
        ring.color = ColorRGBA(r=0.85, g=0.1, b=0.1, a=0.18)
        arr.markers.append(ring)
        # wall (blue cube)
        wall = Marker()
        wall.header = body.header
        wall.ns = "wall"; wall.id = 2; wall.type = Marker.CUBE; wall.action = Marker.ADD
        wall.pose.position = Point(x=self.wc[0], y=self.wc[1], z=self.wc[2])
        wall.pose.orientation.w = 1.0
        wall.scale = Vector3(x=self.we[0], y=self.we[1], z=self.we[2])
        wall.color = ColorRGBA(r=0.15, g=0.35, b=0.85, a=0.55)
        arr.markers.append(wall)
        self.pub_viz.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PerClassObstaclePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
