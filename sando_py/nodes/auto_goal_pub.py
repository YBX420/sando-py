"""Auto terminal-goal publisher for the per-class demo.

sando_node's replan timer starts CANCELLED and only arms on the first term_goal,
so without a goal the drone never plans. This node publishes one PoseStamped on
`term_goal` a few seconds after launch (latched + repeated to beat startup
races), driving start -> goal with no RViz click.

中文说明:
这是演示用的「自动发目标点」节点。规划器(sando_node)的重规划定时器一开始是
停着的,要等收到第一个 term_goal(最终目标)才启动;不发目标飞机就一直不动。

这个节点的作用:启动几秒后(留出各节点起来的时间)自动往 `term_goal` 发一个
目标点,省得每次都要在 RViz 里手点「2D Nav Goal」。它还会在 A、B 两个目标点之间
按固定周期来回切换,让飞机在演示里反复往返,方便观察避障表现。

两个细节(为啥这么写):
  - 话题用 latched(TRANSIENT_LOCAL)QoS:晚订阅的节点也能收到最后一条目标。
  - 每次发目标连发 3 遍:防止规划器订阅还没建好就漏掉第一条(启动竞态)。
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped


class AutoGoalPub(Node):
    """自动发目标点节点:延时启动后在 A、B 两点间周期性来回发 term_goal。"""
    def __init__(self):
        super().__init__("auto_goal_pub")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("goal_x", 10.0)        # end A
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_z", 2.0)
        self.declare_parameter("goal_x2", 0.0)        # end B (start) -> back-and-forth
        self.declare_parameter("goal_y2", 0.0)
        self.declare_parameter("goal_z2", 2.0)
        self.declare_parameter("delay_s", 3.0)
        self.declare_parameter("switch_period_s", 7.0)   # flip the goal every N s
        self.frame = self.get_parameter("frame_id").value
        self.delay = float(self.get_parameter("delay_s").value)
        self.switch_period = float(self.get_parameter("switch_period_s").value)
        self.targets = [
            (float(self.get_parameter("goal_x").value),
             float(self.get_parameter("goal_y").value),
             float(self.get_parameter("goal_z").value)),
            (float(self.get_parameter("goal_x2").value),
             float(self.get_parameter("goal_y2").value),
             float(self.get_parameter("goal_z2").value)),
        ]
        # idx 指向当前要发的目标(0=A, 1=B),_switch 里在 0/1 间翻转
        self.idx = 0

        # TRANSIENT_LOCAL = latched:晚启动的订阅者也能拿到最后一条目标
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PoseStamped, "term_goal", qos)
        # 一次性延时定时器:等 delay 秒(让别的节点先起来)再发第一个目标
        self._delay_timer = self.create_timer(self.delay, self._start)

    # 把 (x,y,z) 目标打包成 PoseStamped 消息
    def _msg(self, g) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = g
        m.pose.orientation.w = 1.0
        return m

    # 延时到了:取消一次性定时器,发出第一个目标(A),再开一个周期定时器负责来回切换
    def _start(self):
        self._delay_timer.cancel()
        self._send()                                       # first goal (end A)
        self.create_timer(self.switch_period, self._switch)   # then flip ends

    # 发当前目标;连发 3 遍是为了赢过启动竞态(订阅可能还没建好)
    def _send(self):
        g = self.targets[self.idx]
        # republish a few times to beat any subscriber-up race
        for _ in range(3):
            self.pub.publish(self._msg(g))
        self.get_logger().info(f"term_goal -> {g}")

    # 周期回调:在 A/B 两点间翻转后重新发,实现往返
    def _switch(self):
        self.idx = (self.idx + 1) % 2                      # back-and-forth
        self._send()


def main(args=None):
    rclpy.init(args=args)
    node = AutoGoalPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
