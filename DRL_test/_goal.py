#!/usr/bin/env python3

# General purpose
import time
import numpy as np

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

from std_msgs.msg import Float32MultiArray

# others

import collections


def _init_goal_subscription(self):
	self.get_logger().info('/goal subscription')
	self.goal_sub = self.create_subscription(
		Float32MultiArray,
		"/goal", 
		self.goal_cb,
		10
	)
	self.goal_pos_x, self.goal_pos_y = 0,0

def goal_cb(self, msg):
	self.get_logger().info('/goal Msg received')
	#self.get_logger().info(str(msg.data))
	#self.get_logger().info(str(msg.data[0]))
	self.get_logger().info('Goal position: x '+str(msg.data[0])+' y '+str(msg.data[1]))
	self.goal_pos_x = msg.data[0]
	self.goal_pos_y = msg.data[1]



#to send a goal msg via shell
"""
ros2 topic pub /goal std_msgs/msg/Float32MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data: [2.0, 3.0]"
"""

