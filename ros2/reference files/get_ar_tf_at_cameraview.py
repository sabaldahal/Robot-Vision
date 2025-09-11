import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion

class EEReader(Node):
    def __init__(self):
        super().__init__('ee_reader')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.get_pose)

    def get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'vx300/base_link', 'vx300/ee_gripper_link', rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            x, y, z = t.x, t.y, t.z
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}, roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        except Exception as e:
            self.get_logger().warn(f"Transform error: {e}")

rclpy.init()
node = EEReader()
rclpy.spin(node)

