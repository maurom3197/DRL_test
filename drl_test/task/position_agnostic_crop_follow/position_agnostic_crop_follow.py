#!/usr/bin/env python3

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from drl_test.task.position_agnostic_crop_follow.position_agnostic_env import DepthAgnosticEnv
from drl_test.utils.tfp_convmix_gaussian_actor import ConvGaussianActor
from gym import spaces

class DRLAgent(DepthAgnosticEnv):
    def __init__(self):
        super().__init__('drl_agent')

        self.declare_parameters(
            namespace="",
            parameters=[
                ("mode", rclpy.Parameter.Type.STRING),
                ("model_path", rclpy.Parameter.Type.STRING),
                ("data_path", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("max_lin_vel", rclpy.Parameter.Type.DOUBLE),
                ("min_lin_vel", rclpy.Parameter.Type.DOUBLE),
                ("max_ang_vel", rclpy.Parameter.Type.DOUBLE),
                ("min_ang_vel", rclpy.Parameter.Type.DOUBLE)
            ],
        )

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )
        self.min_ang_vel = (
            self.get_parameter("min_ang_vel").get_parameter_value().double_value
        )
        self.min_lin_vel = (
            self.get_parameter("min_lin_vel").get_parameter_value().double_value
        )
        self.max_ang_vel = (
            self.get_parameter("max_ang_vel").get_parameter_value().double_value
        )
        self.max_lin_vel = (
            self.get_parameter("max_lin_vel").get_parameter_value().double_value
        )
        self.data_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.control_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )
        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )

        # Instanciate DRL agent
        self.agent = self.instanciate_agent()
        # Init publisher for commands
        qos = QoSProfile(depth=10)
        self._init_drive_camera(qos, self.control_freq)
        
        

    def step(self, ):
        # get depth camera data
        depth_image = self.get_sensor_data()
        state = self.get_observation(self.previous_twist, depth_image)

        # predict action
        action, _ = self.agent(state, True)

        # publish cmd vel
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)

        # update variables
        self.previous_twist = twist


    def _init_drive_camera(self, qos, frequency=15):
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos)

        self.cmd_vel_timer = self.create_timer(
            timer_period_sec = 1/frequency, 
            callback = self.step
            )
        
    def instanciate_agent(self,):
        # Define Action Space
        action = [
            [self.min_lin_vel, self.max_lin_vel],  # x_speed
            [self.min_ang_vel, self.max_ang_vel],  # w_speed
        ]

        low_action = []
        high_action = []
        for i in range(len(action)):
            low_action.append(action[i][0])
            high_action.append(action[i][1])

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        action_space = spaces.Box(
            low=low_action, high=high_action, shape=(2,), dtype=np.float32
        )

        # Define State Space

        state = [
            # [0., 15.], # goal_distance
            # [-math.pi, math.pi], # goal angle or yaw
            [self.min_lin_vel, self.max_lin_vel],  # x_speed
            [self.min_ang_vel, self.max_ang_vel],  # w_speed
        ]

        if self.visual_data == "features":
            for i in range(self.features):
                state = state + [[0.0, self.max_depth]]
        elif self.visual_data == "image":
            self.low_state = np.zeros(
                (self.image_height, self.image_width, self.channels), dtype=np.float32
            )
            self.high_state = self.max_depth * np.ones(
                (self.image_height, self.image_width, self.channels), dtype=np.float32
            )

        if len(state) > 0:
            low_state = []
            high_state = []
            for i in range(len(state)):
                low_state.append(state[i][0])
                high_state.append(state[i][1])

            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        state_shape = self.observation_space.shape
        image_shape= (self.image_height, self.image_width, self.channels)
        action_dim = action_space.high.size
        max_action = action_space.high
        min_action=action_space.low
        agent_units = (256, 256)

        agent = ConvGaussianActor(
            state_shape, image_shape, action_dim, max_action, min_action, squash=True, units=agent_units, name='SAC_Agent')
        # load weights
        agent.load_weights(self.model_path)

        # Check its architecture
        agent.summary()



