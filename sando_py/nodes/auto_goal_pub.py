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
        self.declare_parameter("goal_x", 10.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_z", 2.0)
        self.declare_parameter("delay_s", 3.0)
        self.frame = self.get_parameter("frame_id").value
        self.gx = float(self.get_parameter("goal_x").value)
        self.gy = float(self.get_parameter("goal_y").value)
        self.gz = float(self.get_parameter("goal_z").value)
        self.delay = float(self.get_parameter("delay_s").value)

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PoseStamped, "term_goal", qos)
        self._n = 0
        self._fired = False
        self._delay_timer = self.create_timer(self.delay, self._start)

    def _msg(self) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.pose.position.x = self.gx
        m.pose.position.y = self.gy
        m.pose.position.z = self.gz
        m.pose.orientation.w = 1.0
        return m

    def _start(self):
        if self._fired:
            return
        self._fired = True
        self._delay_timer.cancel()
        self.create_timer(0.5, self._repub)   # republish a few times @2 Hz

    def _repub(self):
        if self._n >= 6:
            return
        self.pub.publish(self._msg())
        self._n += 1
        self.get_logger().info(
            f"published term_goal ({self.gx},{self.gy},{self.gz}) [{self._n}/6]")


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
