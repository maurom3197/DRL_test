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

# others

import collections

from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist


def _init_drive(self, frequency = 20):
	qos = QoSProfile(depth=10)

	self.cmd_vel_pub = self.create_publisher(
		Twist,
		'cmd_vel',
		qos)

	self.cmd_vel_timer = self.create_timer(
		timer_period_sec = 1/frequency, 
		callback = self.cmd_vel_timer_cb
		)


def send_cmd_vel(self,linear_speed, angular_speed):
	twist = Twist() #void instance created
	
	if (linear_speed or angular_speed) is None:
		pass #null action (0,0)
	else:
		twist.linear.x = float(linear_speed) #tf2rl libraries use numpy.float32
		twist.angular.z = float(angular_speed)
	self.cmd_vel_pub.publish(twist)


def cmd_vel_timer_cb(self):
	# 1a) topic che va a leggere goal 
	# 1b) servizio 
	# 2) processa dati sensori 
	# 3) fai inferenza
	# 4) manda comando di velocità

	# compute state
	#state = self.update_observation_lidar()
	#time1 = time.time()
	state = self.update_observation_camera()
	goal = state[0]
	#print('goal distance: ', state[0])
	#print('goal angle: ', state[1])
	depth_image = state[1]

	# compute action
	#self.send_cmd_vel(*self.agent.get_action(np.array(state)))
	self.send_cmd_vel(*self.agent.get_action(goal, depth_image))
	#print(-time1+time.time())
