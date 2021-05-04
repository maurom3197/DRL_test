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

def odometry_sensor_cb(self, msg):
	self.get_logger().debug('/odom Msg received')
	self.odometry_msg = msg

# CAMERA

def _init_camera(self, topic = '/camera/depth/image_raw'):
	self.get_logger().info(topic + ' subscription')
	self.generic_depth_camera_sensor = self.create_subscription(
		Image,
		'/camera/depth/image_raw', 
		self.generic_depth_camera_cb,
		qos_profile_sensor_data
	)
	self.bridge = CvBridge()

def generic_depth_camera_cb(self, msg):
	self.get_logger().debug('/camera/depth/image_raw Msg received')
	depth_image_raw = np.zeros((120,160), np.uint8)
	depth_image_raw = self.bridge.imgmsg_to_cv2(msg, '32FC1')
	
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
	goal_distance, goal_angle, pos_x, pos_y, yaw = process_odom(self.odometry_msg, self.goal_pos_x, self.goal_pos_y)

	return ([goal_distance] + \
			[goal_angle] 	+ \
			processed_lidar)

def update_observation_camera(self, image_height = 120, image_width = 160):
	processed_depth_image = process_depth_image(self.generic_depth_camera_img,\
		image_height, image_width)

	# x, y, yaw = pose_2_xyyaw(self.odometry_msg)
	# goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
	# goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
	goal_distance, goal_angle, pos_x, pos_y, yaw = process_odom(self.odometry_msg, self.goal_pos_x, self.goal_pos_y)
	goal_info = np.array([goal_distance, goal_angle], dtype=np.float32)
	goal_info = tf.convert_to_tensor(goal_info)

	return (goal_info, processed_depth_image)

def laserscan_2_n_points_list(laserscan_data, n_points = 36):
	n_points_list = []
	len_laserscan_data = len(laserscan_data.ranges)
	for index in range(n_points):
		n_points_list.append(\
			laserscan_data.ranges[int(index*len_laserscan_data/n_points)]
			)

	return n_points_list #type: list (of float)

def process_odom(odom_msg, goal_pose_x, goal_pose_y):

	pos_x = odom_msg.pose.pose.position.x
	pos_y = odom_msg.pose.pose.position.y
	_,_,yaw = euler_from_quaternion(odom_msg.pose.pose.orientation)

	goal_distance = math.sqrt(
		(goal_pose_x-pos_x)**2
		+ (goal_pose_y-pos_y)**2)

	path_theta = math.atan2(
		goal_pose_y-pos_y,
		goal_pose_x-pos_x)

	goal_angle = path_theta - yaw

	if goal_angle > math.pi:
		goal_angle -= 2 * math.pi

	elif goal_angle < -math.pi:
		goal_angle += 2 * math.pi

	return goal_distance, goal_angle, pos_x, pos_y, yaw

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


def clean_laserscan(laserscan_data, laser_range = 5):
	# Takes only sensed measurements
	for i in range(359):
		if laserscan_data.ranges[i] == float('Inf'):
			laserscan_data.ranges[i] = laser_range #set range to max
		elif np.isnan(laserscan_data.ranges[i]):
			laserscan_data.ranges[i] = 0.0 #set range to 0
		else:
			pass # leave range as it is
	return laserscan_data

def process_depth_image(image, height, width):
	img = np.array(image, dtype= np.float32)
	cutoff = 8
	#check crop is performed correctly
	#img = tf.convert_to_tensor(self.depth_image_raw, dtype=tf.float32)
	#img = img.reshape(240,320,1)
	img = tf.reshape(img, [height,width,1])
	#width =304
	#height = 228
	#h_off = int((240-height)*0.5)
	#w_off = int((320-width)*0.5)
	#img_crop = tf.image.crop_to_bounding_box(img,h_off,w_off,height,width)
	img_resize = tf.image.resize(img,[60,80])
	depth_image = tf.reshape(img_resize, [60,80])
	depth_image = np.array(depth_image, dtype= np.float32)
	depth_image = depth_rescale(depth_image, cutoff)
	final_image_size = depth_image.shape
	return depth_image

def depth_rescale(img, cutoff):
	#Useful to turn the background into black into the depth images.
	w,h = img.shape
	#new_img = np.zeros([w,h,3])
	img = img.flatten()
	img[np.isnan(img)] = cutoff
	img[img>cutoff] = cutoff
	img = img.reshape([w,h])
	#img = img/cutoff
	#img_visual = 255*(self.depth_image_raw/cutoff)
	img = np.array(img, dtype=np.float32)
	return img 
