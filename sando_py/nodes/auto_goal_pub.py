"""Auto terminal-goal publisher for the per-class demo.

sando_node's replan timer starts CANCELLED and only arms on the first term_goal,
so without a goal the drone never plans. This node publishes one PoseStamped on
`term_goal` a few seconds after launch (latched + repeated to beat startup
races), driving start -> goal with no RViz click.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped


class AutoGoalPub(Node):
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
        self.idx = 0

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PoseStamped, "term_goal", qos)
        self._delay_timer = self.create_timer(self.delay, self._start)

    def _msg(self, g) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = g
        m.pose.orientation.w = 1.0
        return m

    def _start(self):
        self._delay_timer.cancel()
        self._send()                                       # first goal (end A)
        self.create_timer(self.switch_period, self._switch)   # then flip ends

    def _send(self):
        g = self.targets[self.idx]
        # republish a few times to beat any subscriber-up race
        for _ in range(3):
            self.pub.publish(self._msg(g))
        self.get_logger().info(f"term_goal -> {g}")

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
