#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VisualizationNode(Node):
    """
    Subscribes to /perception/image_out and displays the frames in an OpenCV window.
    """
    def __init__(self):
        super().__init__('visualization_node')
        self.subscription = self.create_subscription(
            Image,
            '/perception/image_out',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info('VisualizationNode started.')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Display in a CV window
        cv2.imshow('Perception Visualization', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
