#!/usr/bin/env python3

# General purpose
import time
import numpy as np

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs

class RealsensePy(Node):

	def __init__(self):
		super().__init__("realsensepy")
		rclpy.logging.set_logger_level('realsensepy', 10)

		# Configure depth and color streams
		self.pipeline = rs.pipeline()
		self.config = rs.config()

		# Get device product line for setting a supporting resolution
		self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
		self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
		self.device = self.pipeline_profile.get_device()
		self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

		self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

		# Start streaming
		self.pipeline.start(self.config)

		# Create an align object
		# rs.align allows us to perform alignment of depth frames to others frames
		# The "align_to" is the stream type to which we plan to align depth frames.
		self.align_to = rs.stream.color
		self.align = rs.align(self.align_to)
		
		# Create Publishers
		self.color_topic = '/camera/rgb/image_raw'
		self.depth_topic = '/camera/depth/image_raw'
		#self.color_publisher_ = self.create_publisher(Image, self.color_topic, 10)
		#self.get_logger().info('start publishing on ' + self.color_topic)
		self.depth_publisher_ = self.create_publisher(Image, self.depth_topic, 10)
		self.get_logger().info('start publishing on ' + self.depth_topic)
		self.bridge = CvBridge()
		self.run()

	def run(self):

		try:
			while True:
				# Wait for a coherent pair of frames: depth and color
				frames = self.pipeline.wait_for_frames()
				# Align the depth frame to color frame
				aligned_frames = self.align.process(frames)

				depth_frame = aligned_frames.get_depth_frame()
				color_frame = aligned_frames.get_color_frame()
				if not depth_frame or not color_frame:
				    continue

				# Convert images to numpy arrays
				depth_image = np.asanyarray(depth_frame.get_data())
				#color_image = np.asanyarray(color_frame.get_data())

				# Publish images on ros topics
				#self.color_publisher_.publish(self.bridge.cv2_to_imgmsg(color_image, "bgr8"))
				#self.get_logger().info('Publishing an RGB image')
				self.depth_publisher_.publish(self.bridge.cv2_to_imgmsg(depth_image, "16UC1"))
				#self.get_logger().info('Publishing a depth image')

				# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
				#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

				#depth_colormap_dim = depth_colormap.shape
				#color_colormap_dim = color_image.shape

				#If depth and color resolutions are different, resize color image to match depth image for display
				#if depth_colormap_dim != color_colormap_dim:
				#   resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
				#   images = np.hstack((resized_color_image, depth_colormap))
				#else:
				#   images = np.hstack((color_image, depth_colormap))

				#Show images
				#cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
				#cv2.imshow('RealSense', images)
				#cv2.waitKey(1)

			
		finally:

		    # Stop streaming
		    self.pipeline.stop()

def main(args=None):
	rclpy.init()
	realsense_py = RealsensePy()
	realsense_py.get_logger().info('Spinning')
	rclpy.spin(realsense_py)

	realsense_py.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
