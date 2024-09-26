# Import the FlagDetector class from FlagDetector_class.py
from FlagDetector_class import FlagDetector
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, \
    QoSDurabilityPolicy

from autositter_offboard_msgs.msg import FlagReport

class FlagDetectorNode(Node):
    def __init__(self):
        super().__init__('flag_detector')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Declare parameters without default values
        self.declare_parameter('color_filters')
        self.declare_parameter('rot_compensation_mode')
        self.declare_parameter('safe_column_width_ratio')
        self.declare_parameter('reference_image_path')

        # Get parameters with correct data type enforcement
        self.color_filters = self.get_parameter('color_filters').get_value()
        self.rot_compensation_mode = self.get_parameter('rot_compensation_mode').as_string()
        self.safe_column_width_ratio = self.get_parameter('safe_column_width_ratio').as_double()
        self.reference_image_path = self.get_parameter('reference_image_path').as_string()

        # Log parameters for debugging
        # self.get_logger().info(f"Color Filters: {self.color_filters}")
        # self.get_logger().info(f"Rotation Compensation Mode: {self.rot_compensation_mode}")
        # self.get_logger().info(f"Safe Column Width Ratio: {self.safe_column_width_ratio}")
        # self.get_logger().info(f"Reference Image Path: {self.reference_image_path}")

        # Initialize the FlagDetector class with the parameters
        self.flag_detector = FlagDetector(
            reference_path=self.reference_image_path,
            hsv_ranges=self.color_filters,
            safe_column_width_ratio=self.safe_column_width_ratio
        )

        # Set up the publisher and subscriber
        self.flag_report_pub = self.create_publisher(
            FlagReport,
            'FlagReport',
            qos_profile)
        self.webcam_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.webcam_callback,
            qos_profile)

        self.br = CvBridge()

    def webcam_callback(self, frame):
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        self.current_frame = self.br.imgmsg_to_cv2(frame, desired_encoding='bgr8')

        # Process the frame using the FlagDetector class
        processed_frame = self.flag_detector.process_frame(self.current_frame, mode=self.rot_compensation_mode)

        # Optionally, display the processed frame (remove if running headless)
        cv.imshow('Processed Frame', processed_frame)
        cv.waitKey(1)

        # Create and publish FlagReport message based on detection
        flag_report_msg = FlagReport()
        flag_report_msg.is_flag = self.flag_detector.is_flag

        # Convert error tuple to appropriate message fields
        flag_report_msg.error_x = float(self.flag_detector.error[0])
        flag_report_msg.error_y = float(self.flag_detector.error[1])

        flag_report_msg.aiming = float(self.flag_detector.aiming)
        flag_report_msg.detected_num = int(self.flag_detector.detected_num)
        flag_report_msg.is_within_safe_column = self.flag_detector.is_within_safe_column

        self.flag_report_pub.publish(flag_report_msg)

def main(args=None):
    rclpy.init(args=args)

    detector = FlagDetectorNode()

    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
