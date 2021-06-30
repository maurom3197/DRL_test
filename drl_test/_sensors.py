#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import tensorflow as tf
import math

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others
from numpy import savetxt
import collections
import cv2
from cv_bridge import CvBridge


# ODOMETRY

def _init_odometry(self):
	self.get_logger().info('/odom subscription')
	self.odometry_sensor_sub = self.create_subscription(
		Odometry,
		"/odom", 
		self.odometry_sensor_cb,
		10
	)
	self.init_yaw = None

def odometry_sensor_cb(self, msg):
	self.get_logger().debug('/odom Msg received')
	self.odometry_msg = msg

# CAMERA

def _init_camera(self, topic = '/camera/depth/image_raw'):
	self.get_logger().info(topic + ' subscription')
	self.generic_depth_camera_sensor = self.create_subscription(
		Image,
		topic, 
		self.generic_depth_camera_cb,
		qos_profile_sensor_data
	)
	self.bridge = CvBridge()

def generic_depth_camera_cb(self, msg, encoding = '16UC1'):
	self.get_logger().debug('/camera/depth/image_raw Msg received')
	depth_image_raw = np.zeros((self.image_height,self.image_width), np.uint8)
	depth_image_raw = self.bridge.imgmsg_to_cv2(msg, encoding)
	
	self.generic_depth_camera_msg = msg
	self.generic_depth_camera_img = depth_image_raw


# LIDAR

def _init_lidar(self):
	self.get_logger().info('/scan subscription')
	self.generic_laser_scan_sensor_sub = self.create_subscription(
		LaserScan,
		"/scan", 
		self.generic_laser_scan_cb,
		qos_profile_sensor_data
	)

def generic_laser_scan_cb(self, msg):
	self.get_logger().debug('/scan Msg received')
	self.laser_scan_msg = msg


######################################################################
def update_observation_lidar(self, lidar_points = 36):
	processed_lidar = laserscan_2_n_points_list(
		clean_laserscan(self.laser_scan_msg),\
		lidar_points
	)
	#x, y, yaw = pose_2_xyyaw(self.odometry_msg)
	#goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
	#goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
	goal_distance, goal_angle, pos_x, pos_y, yaw, init_yaw = process_odom(self.odometry_msg, self.goal_pos_x, self.goal_pos_y, self.init_yaw)

	return ([goal_distance] + \
			[goal_angle] 	+ \
			processed_lidar)

def update_observation_camera(self):
	processed_depth_image = process_depth_image(self.generic_depth_camera_img,\
		self.image_height, self.image_width)

	# x, y, yaw = pose_2_xyyaw(self.odometry_msg)
	# goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
	# goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
	goal_distance, goal_angle, pos_x, pos_y, yaw, init_yaw = process_odom(self.odometry_msg, self.goal_pos_x, self.goal_pos_y, self.init_yaw)
	self.init_yaw = init_yaw
	goal_info = np.array([goal_distance, goal_angle], dtype=np.float32)
	goal_info = tf.convert_to_tensor(goal_info)

	return (goal_info, processed_depth_image)

def clean_laserscan(laserscan_data, laser_range = 5.0):
	# Takes only sensed measurements
	len_laserscan_data = len(laserscan_data.ranges)
	for i in range(len_laserscan_data):
		if laserscan_data.ranges[i] == float('Inf'):
			laserscan_data.ranges[i] = laser_range #set range to max
		elif np.isnan(laserscan_data.ranges[i]):
			laserscan_data.ranges[i] = 0.0 #set range to 0
		else:
			pass # leave range as it is
	return laserscan_data
	
def laserscan_2_n_points_list(laserscan_data, n_points = 36):
	n_points_list = []
	len_laserscan_data = len(laserscan_data.ranges)
	#print('len laser scan ', len_laserscan_data)
	for index in range(n_points):
		points_index = int(index*len_laserscan_data/n_points)
		#print('point index ', points_index)
		n_points_list.append(\
			laserscan_data.ranges[points_index]
			)

	return n_points_list #type: list (of float)

def process_odom(odom_msg, goal_pose_x, goal_pose_y, init_yaw):

	pos_x = odom_msg.pose.pose.position.x
	pos_y = odom_msg.pose.pose.position.y
	_,_,yaw = euler_from_quaternion(odom_msg.pose.pose.orientation)

	if init_yaw == None:
		init_yaw = yaw
		print('Initial ref YAW is: ', yaw)

	#print('corrected yaw', yaw - init_yaw)

	goal_distance = math.sqrt(
		(goal_pose_x-pos_x)**2
		+ (goal_pose_y-pos_y)**2)

	path_theta = math.atan2(
		goal_pose_y-pos_y,
		goal_pose_x-pos_x)

	#print('path theta', path_theta)
	goal_angle = path_theta - yaw

	if goal_angle > math.pi:
		goal_angle -= 2 * math.pi

	elif goal_angle < -math.pi:
		goal_angle += 2 * math.pi

	return goal_distance, goal_angle, pos_x, pos_y, yaw, init_yaw

def euler_from_quaternion(quat):
	"""
	Converts quaternion (w in last place) to euler roll, pitch, yaw
	quat = [x, y, z, w]
	"""
	x = quat.x
	y = quat.y
	z = quat.z
	w = quat.w

	sinr_cosp = 2 * (w*x + y*z)
	cosr_cosp = 1 - 2*(x*x + y*y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2 * (w*y - z*x)
	pitch = np.arcsin(sinp)

	siny_cosp = 2 * (w*z + x*y)
	cosy_cosp = 1 - 2 * (y*y + z*z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw


def process_depth_image(image, height, width):
	img = np.array(image, dtype= np.float32)

	# SHOW IMAGE
	#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)
	#cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
	#cv2.imshow('RealSense', depth_colormap)
	#cv2.waitKey(1)

	cutoff = 8 # mm for the hardware camera sensor, m in simulation
	img = tf.reshape(img, [height,width,1])

	# FOR CROPPING IMAGE
	#width =304
	#height = 228
	#h_off = int((240-height)*0.5)
	#w_off = int((320-width)*0.5)
	#img_crop = tf.image.crop_to_bounding_box(img,h_off,w_off,height,width)

	# RESIZE IMAGE to [60,80]
	img_resize = tf.image.resize(img,[60,80])
	depth_image = tf.reshape(img_resize, [60,80])
	depth_image = np.array(depth_image, dtype= np.float32)
	#savetxt('/home/maurom/mauro_local_ws/depth_images/REALtxt_depth_image_resize.csv', depth_image, delimiter=',')
	depth_image = depth_rescale(depth_image, cutoff)
	image_size = depth_image.shape
	#print(depth_image)
	#savetxt('/home/maurom/mauro_local_ws/depth_images/REALtxt_depth_image_processed.csv', depth_image, delimiter=',')
	return depth_image

def depth_rescale(img, cutoff):
	#Useful to turn the background into black into the depth images.
	w,h = img.shape
	#new_img = np.zeros([w,h,3])
	img = img.flatten()
	img[np.isnan(img)] = cutoff
	img[img>cutoff] = cutoff
	img = img.reshape([w,h])
	img = img/cutoff

	return img 
