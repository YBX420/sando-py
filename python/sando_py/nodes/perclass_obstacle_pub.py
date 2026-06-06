"""Demo obstacle publisher for the per-class RViz demo.

Publishes two obstacles on `predicted_trajs` (dynus_interfaces/DynTraj) so the
planner sees them, plus its OWN visualization MarkerArray on `perclass_obstacles`
(sando_node draws no obstacle markers):

  - HUMAN  (id 100, HARD): a person crossing the drone's path, analytic motion.
  - WALL   (id 200, SOFT): a static thin box the drone may graze.

The id range is the swappable class tag the planner reads
(200<=id<300 -> wall/SOFT, else human/HARD).

中文说明:
这是「按类别避障」演示用的障碍发布节点。它在场景里造两个假障碍喂给规划器,
专门用来检验 per-class 避障逻辑——同样是障碍,人和墙走两套不同的处理。

它发两个话题:
  - `predicted_trajs`(dynus_interfaces/DynTraj):规划器真正读的障碍信息,
    包含解析运动公式和包围盒,规划器据此预测障碍未来位置去避让。
  - `perclass_obstacles`(MarkerArray):仅供 RViz 显示的彩色 marker
    (sando_node 自己不画障碍,所以这里自己画)。

两个障碍:
  - 人(id=100,HARD 硬约束):横穿飞机航线、左右来回走;规划器对人用
    凸包 + ALM(增广拉格朗日)的「时空硬约束」,要求绝对不许撞。
  - 墙(id=200,SOFT 软场):一个固定的薄方块;规划器对墙用 EGO 那种
    「软的排斥力场」,允许蹭一下、代价高一点而已,不是硬禁区。

关键约定:障碍的 id 数值段就是「类别标签」,规划器靠它分流——
200<=id<300 当成墙(SOFT),其它当成人(HARD)。改 id 就能换类别。
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from dynus_interfaces.msg import DynTraj


class PerClassObstaclePub(Node):
    """造两个假障碍(来回走的人 + 静止的墙)并以 10Hz 发布,给 per-class 避障演示用。"""
    def __init__(self):
        super().__init__("perclass_obstacle_pub")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("human_x", 5.0)
        self.declare_parameter("human_amp", 2.5)        # y oscillation amplitude (m)
        self.declare_parameter("human_period", 6.0)     # back-and-forth period (s)
        self.declare_parameter("human_z", 1.0)
        self.declare_parameter("wall_center", [5.0, 1.2, 2.0])
        self.declare_parameter("wall_extents", [2.0, 0.4, 3.0])
        self.frame = self.get_parameter("frame_id").value
        self.hx = float(self.get_parameter("human_x").value)
        self.amp = float(self.get_parameter("human_amp").value)
        self.period = float(self.get_parameter("human_period").value)
        # w = 角频率(2π/周期),后面用 sin(w*t) 让人左右来回走
        self.w = 2.0 * math.pi / max(self.period, 1e-3)
        self.hz = float(self.get_parameter("human_z").value)
        # wall_center=墙中心坐标, wall_extents=墙在 xyz 三个方向的边长
        self.wc = [float(v) for v in self.get_parameter("wall_center").value]
        self.we = [float(v) for v in self.get_parameter("wall_extents").value]
        # 人的包围盒(xyz 边长),0.6x0.6x1.8 大致是一个站着的人
        self.human_bbox = [0.6, 0.6, 1.8]

        # t0=节点启动时刻,作为正弦运动的时间起点
        self.t0 = self._now()
        self.pub_traj = self.create_publisher(DynTraj, "predicted_trajs", 10)
        self.pub_viz = self.create_publisher(MarkerArray, "perclass_obstacles", 10)
        self.create_timer(0.1, self._tick)   # 10 Hz

    # 取当前 ROS 时间(秒,带纳秒小数),后面用绝对时间算正弦位置
    def _now(self) -> float:
        t = self.get_clock().now().to_msg()
        return float(t.sec) + 1e-9 * float(t.nanosec)

    # 人当前位置:x、z 固定,y 按 sin 来回摆(横穿飞机航线)
    def _human_pos(self, t: float):
        return [self.hx, self.amp * math.sin(self.w * (t - self.t0)), self.hz]

    # 组装一条 DynTraj 障碍消息;function 是 xyz 三个「解析运动公式」字符串,
    # velocity 是对应的速度公式,规划器会按这些公式预测障碍未来轨迹。
    def _dyntraj(self, tid, bbox, fx, fy, fz, vx, vy, vz, pos):
        m = DynTraj()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.id = int(tid)
        m.is_agent = False
        m.bbox = [float(b) for b in bbox]
        # mode="analytic":告诉规划器这条轨迹有闭式公式,可直接代入时间 t 求值
        m.mode = "analytic"
        m.function = [fx, fy, fz]
        m.velocity = [vx, vy, vz]
        m.pos = Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.goal = [float(pos[0]), float(pos[1]), float(pos[2])]
        return m

    # 10Hz 主循环:算出人当前位置,把人和墙两条 DynTraj 发给规划器,再发可视化 marker
    def _tick(self):
        now = self._now()
        hp = self._human_pos(now)
        # HUMAN (HARD, id 100): analytic back-and-forth in y (sin), t = absolute ROS sec.
        # velocity is the exact analytic derivative so the space-time ALM tracks it.
        # 中文:人(硬约束,id=100)。y 方向用 sin 来回走,公式里的 t 是绝对 ROS 秒。
        # 速度公式给的是 sin 的精确导数(amp*w*cos),这样规划器的「时空 ALM 硬约束」
        # 才能算准人下一刻在哪、提前躲开。
        human = self._dyntraj(
            100, self.human_bbox,
            f"{self.hx}", f"{self.amp}*sin({self.w}*(t - {self.t0}))", f"{self.hz}",
            "0.0", f"{self.amp * self.w}*cos({self.w}*(t - {self.t0}))", "0.0", hp)
        # WALL (SOFT, id 200): static box
        # 中文:墙(软约束,id=200)。位置固定、速度全 0,就是个不动的方块。
        wall = self._dyntraj(
            200, self.we,
            f"{self.wc[0]}", f"{self.wc[1]}", f"{self.wc[2]}",
            "0.0", "0.0", "0.0", self.wc)
        self.pub_traj.publish(human)
        self.pub_traj.publish(wall)
        self._publish_markers(hp)

    # 发布 RViz 可视化:人(红球)+ 人的安全壳(半透明红球)+ 墙(蓝方块)
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
        # 中文:人的「安全壳」——半透明红球,半径 = 人半径 + 安全距离 0.8m。
        # 这个壳就是规划器对人要求的最小避让距离,飞机不该钻进这层壳里。
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
