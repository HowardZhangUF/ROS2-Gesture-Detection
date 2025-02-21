#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class CameraNode(Node):
    """
    A simple ROS 2 camera node that publishes images from a webcam or directory.
    """

    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('use_webcam', True)
        self.declare_parameter('image_directory', '')
        self.declare_parameter('publish_topic', '/camera/image_raw')
        self.declare_parameter('fps', 10)

        # Get parameters
        self.use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        self.image_directory = self.get_parameter('image_directory').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value

        # Set up publishing
        self.publisher_ = self.create_publisher(Image, self.publish_topic, 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / self.fps, self.publish_image)

        # Initialize webcam or image list
        self.cap = None
        self.image_list = []
        self.image_index = 0

        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error('Could not open webcam. Check your device.')
                rclpy.shutdown()
        else:
            if os.path.isdir(self.image_directory):
                self.image_list = sorted([
                    os.path.join(self.image_directory, f)
                    for f in os.listdir(self.image_directory)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                if not self.image_list:
                    self.get_logger().error('No images found in the specified directory.')
                    rclpy.shutdown()
            else:
                self.get_logger().error('Invalid image directory provided.')
                rclpy.shutdown()

        self.get_logger().info('Camera node started. Publishing images on: {}'.format(self.publish_topic))

    def publish_image(self):
        """
        Publishes an image either from a webcam or a directory.
        """
        if self.use_webcam:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('Failed to capture image from webcam.')
                return
        else:
            if not self.image_list:
                self.get_logger().error('No images to publish.')
                return
            frame = cv2.imread(self.image_list[self.image_index])
            self.image_index = (self.image_index + 1) % len(self.image_list)

        # Convert and publish the image
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Image published.')
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')

    def destroy_node(self):
        """
        Ensure resources are cleaned up properly.
        """
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Camera node shutting down.')
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
