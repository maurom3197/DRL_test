#!/usr/bin/env python3

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from drl_test.task.position_agnostic_crop_follow.position_agnostic_env import DepthAgnosticEnv
from drl_test.utils.tfp_convmix_gaussian_actor import ConvGaussianActor
from gym import spaces

from tf2rl.algos.sac import SAC

class DRLAgent(DepthAgnosticEnv):
    def __init__(self):
        super().__init__()

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
        self.output_dir = "/root/gym_ws/src/DRL_test/output_dir"
        self.previous_twist = Twist()
        self.agent = self.instanciate_agent()
        # Init publisher for commands
        qos = QoSProfile(depth=10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        #self._init_drive_camera(qos, self.control_freq)
        
        
    def step(self, ):
        self.get_logger().debug("Stepping...")

        if self.agent is None:
            self.get_logger().debug("Waiting for agent to be ready...")
            return
        
        # get depth camera data
        self.spin_sensors_callbacks()
        self.get_logger().debug("getting sensor data...")
        depth_image = self.get_sensor_data()
        state = self.get_observation(self.previous_twist, depth_image)

        # predict action
        self.get_logger().debug("predicting action...")

        #action, _ = self.agent(state, True)
        action = self.agent.get_action(state, test=True)
        action = action.numpy().flatten()

        # publish cmd vel
        self.get_logger().debug("publishing action...")
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)
        rclpy.spin_once(self)

        # update variables
        self.previous_twist = twist


    def _init_drive_camera(self, qos):
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos)

        # self.cmd_vel_timer = self.create_timer(
        #     timer_period_sec = 1/frequency, 
        #     callback = self.step
        #     )
        
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

        self.action_space = spaces.Box(
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

        # state_shape = self.observation_space.shape
        # image_shape= (self.image_height, self.image_width, self.channels)
        # action_dim = action_space.high.size
        # max_action = action_space.high
        # min_action=action_space.low
        # agent_units = (256, 256)

        self.get_logger().debug("Defining SAC Policy Agent...")
        policy = SAC(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    image_shape=(self.image_height, self.image_width, 1),
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    network="conv",
                    tau=5e-3,
                    alpha=0.2
                )
                
        #agent = ConvGaussianActor(
        #    state_shape, image_shape, action_dim, max_action, min_action, squash=True, units=agent_units, name='SAC_Agent')
        # load weights
        #agent.load_weights(self.model_path)
        self.get_logger().debug("Settting checkpoint...")
        self.set_checkpoint(policy, self.model_path)

        # Check its architecture
        #agent.summary()
        return policy

    def set_checkpoint(self, policy, model_dir):
        # Save and restore model
        checkpoint = tf.train.Checkpoint(policy=policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.output_dir, max_to_keep=5)

        if model_dir is not None:
            self.get_logger().debug("Model directory path {}".format(model_dir))
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            checkpoint.restore(self._latest_path_ckpt)
            self.get_logger().info("Restored {}".format(self._latest_path_ckpt))
    
    def execute(self,):
        while True:
            time.sleep(1/self.control_freq)
            self.step()


