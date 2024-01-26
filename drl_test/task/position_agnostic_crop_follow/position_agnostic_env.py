#!/usr/bin/env python3

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

import rclpy
from rclpy.node import Node

import numpy as np
from numpy import savetxt
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from drl_test.sensors import Sensors

class DepthAgnosticEnv(Node):
    def __init__(self):
        super().__init__("depth_agnostic_env")
        #rclpy.logging.set_logger_level('depth_agnostic_env', 10)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("sensor", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("visual_data", rclpy.Parameter.Type.STRING),
                ("features", rclpy.Parameter.Type.INTEGER),
                ("channels", rclpy.Parameter.Type.INTEGER),
                ("depth_param.width", rclpy.Parameter.Type.INTEGER),
                ("depth_param.height", rclpy.Parameter.Type.INTEGER),
                ("depth_param.dist_cutoff", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
            ],
        )

        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.visual_data = (
            self.get_parameter("visual_data").get_parameter_value().string_value
        )
        self.features = (
            self.get_parameter("features").get_parameter_value().integer_value
        )
        self.channels = (
            self.get_parameter("channels").get_parameter_value().integer_value
        )
        self.image_width = (
            self.get_parameter("depth_param.width").get_parameter_value().integer_value
        )
        self.image_height = (
            self.get_parameter("depth_param.height").get_parameter_value().integer_value
        )
        self.max_depth = (
            self.get_parameter("depth_param.dist_cutoff").get_parameter_value().double_value
        )
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", qos)

    def spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            empty_measurements = [ k for k, v in self.sensors.sensor_msg.items() if v is None]
            self.get_logger().debug(f"empty_measurements: {empty_measurements}")
            rclpy.spin_once(self)
            self.get_logger().debug("spin once ...")
        self.get_logger().debug("spin sensor callback complete ...")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def get_sensor_data(self):
        """ """
        sensor_data = {}
        sensor_data["depth"] = self.sensors.get_depth()

        if sensor_data["depth"] is None:
            sensor_data["depth"] = (
                np.ones((self.image_height, self.image_width, 1)) * self.max_depth
            )

        self.get_logger().debug("processing odom...")
        depth_image = sensor_data["depth"]

        return depth_image
    
    def get_observation(self, twist, depth_image):
        """ """
        # flattened depth image
        if self.visual_data == "features":
            features = depth_image.flatten()

        # previous velocity state
        v = twist.linear.x
        w = twist.angular.z
        vel = np.array([v, w], dtype=np.float32)
        state = np.concatenate((vel, features))
        state = tf.constant(state, dtype=tf.float32)
        state = tf.expand_dims(state, 0) 
        return state