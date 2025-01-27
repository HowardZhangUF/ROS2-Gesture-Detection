#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2


from .mediapipe_rov import rov, ROVAction

class PerceptionNode(Node):
    """
    Subscribes to /camera/image_raw, runs the 'rov' pipeline,
    publishes overlayed image on /perception/image_out,
    and also publishes the action string and perception mode.
    """

    def __init__(self):
        super().__init__('perception_node')

        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        
        self.image_pub = self.create_publisher(Image, '/perception/image_out', 10)
        self.action_pub = self.create_publisher(String, '/perception/action', 10)
        self.mode_pub   = self.create_publisher(String, '/perception/mode', 10)

        
        self.bridge = CvBridge()

        
        rov_action = ROVAction()
        self.bluerov = rov(camera_id=0, action=rov_action)

        

        self.get_logger().info("Perception Node initialized.")

    def image_callback(self, msg):
        """
        Receives images from /camera/image_raw, passes them to self.bluerov,
        runs detection, updates tasks, then publishes the results.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        
        self.bluerov.image_buffer = frame

        # Manually update counters like get_camera_image() does
        self.bluerov.frame_counter += 1
        if self.bluerov.frame_counter % self.bluerov.CTRL_INT == 0:
            self.bluerov.perception_counter += 1

        # Run the perception pipeline
        self.bluerov.get_perception()
        action_str = self.bluerov.update_task()  # we modified update_task() to return a string

        # Publish the action string if it exists
        if action_str is not None:
            self.action_pub.publish(String(data=action_str))

        # Publish the current perception_mode
        mode_str = self.bluerov.perception_mode
        self.mode_pub.publish(String(data=mode_str))

        # Create an overlay to publish
        overlay = self.bluerov.visualize_perception()
        if overlay is not None:
            out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            self.image_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
