#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class RobotNavigation(Node):
	from _sensors import _init_odometry, odometry_sensor_cb
	from _sensors import _init_camera, generic_depth_camera_cb
	from _sensors import _init_lidar, generic_laser_scan_cb
	from _sensors import update_observation

	from _drive import _init_drive, send_cmd_vel, cmd_vel_timer_cb

	from _goal import _init_goal_subscription, goal_cb

	from _agent import _init_agent

	def __init__(self):
		super().__init__("drl_navigation")
		rclpy.logging.set_logger_level('pic4rl', 10)

		self._init_odometry()
		# msg --> self.odometry_msg

		#self._init_camera() 
		# msg --> self.generic_depth_camera_img

		self._init_lidar() 
		# msg --> self.laser_scan_msg

		# initialize also publisher for cmd_vel
		# with a timer, to set a control frequency
		self._init_goal_subscription()

		while True:
			try:
				rclpy.spin_once(self, timeout_sec=5)
				self.laser_scan_msg
				self.odometry_msg
				break
			except:
				pass


		self._init_drive(frequency = 20)


		self._init_agent()

		# agent init
		#





def main(args=None):
	try:
		rclpy.init()
		robot_navigation = RobotNavigation()
		robot_navigation.get_logger().info('Spinning')
		rclpy.spin(robot_navigation)
	finally:
		robot_navigation.destroy_node()
		rclpy.shutdown()



if __name__ == '__main__':
	main()
