#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

import json
import numpy as np
import random
import sys
import time
import math

from NeuralNetworks import  ActorNetwork, ActorCNNetwork


def _init_agent_lidar(self):
	self.agent = DDPGLidarAgent(
			state_size = 38, 
			action_size = 2, 
			max_linear_vel = 0.5,
			max_angular_vel = 1.0,
			load = True
			)

def _init_agent_camera(self):
	self.agent = DDPGVisualAgent(
			state_size = 3, 
			action_size = 2, 
			max_linear_vel = 0.5,
			max_angular_vel = 1.0,
			load = True
			)


class DDPGLidarAgent:

	def __init__(self, state_size, action_size = 2, max_linear_vel = 0.5, max_angular_vel = 1.0, load = True):


		# State size and action size
		self.state_size = state_size 
		self.action_size = action_size 
		self.max_linear_vel = max_linear_vel
		self.max_angular_vel = max_angular_vel

		self.actor = ActorNetwork(self.state_size, self.max_linear_vel, self.max_angular_vel, lr = 0.00025, fc1_dims = 256, fc2_dims = 128, fc3_dims = 128, name = 'actor')

		#Load Models

		self.load = load
		self.load_episode = 2800

		if self.load:
			#actor_dir_path = os.path.join(
			#	self.actor.model_dir_path,
			#	'actor_stage1_episode'+str(self.load_episode)+'.h5')
			model_dir_path = os.path.dirname(os.path.realpath(__file__))
			actor_dir_path = model_dir_path + "/agent_weights/jackal/lidar/actor_weights_episode" + str(self.load_episode)+ ".h5"
			self.actor.load_weights(actor_dir_path)


	def get_action(self, state):
		if state[0] < 0.15:
			print('Goal reached...')
			return np.array([0.0, 0.0], dtype= np.float32)

		pred_action = self.actor(state.reshape(1, len(state)))
		pred_action = tf.reshape(pred_action, [2,])
		print("pred_action:", pred_action)
		return pred_action
		#return np.array([0.0, 0.0], dtype= np.float32)

class DDPGVisualAgent:

	def __init__(self, state_size, image_height=60, image_width=80, action_size = 2, max_linear_vel = 0.5, max_angular_vel = 1.0, load = True):


		# State size and action size
		self.state_size = state_size 
		self.action_size = action_size 
		self.max_linear_vel = max_linear_vel
		self.max_angular_vel = max_angular_vel
		self.image_height = image_height
		self.image_width = image_width
		self.actor = ActorCNNetwork(max_linear_velocity = self.max_linear_vel, max_angular_velocity = self.max_angular_vel, lr = 0.0001, name = 'actor')

		#Load Models

		self.load = load
		self.load_episode = 3200

		if self.load:
			#actor_dir_path = os.path.join(
			#	self.actor.model_dir_path,
			#	'actor_stage1_episode'+str(self.load_episode)+'.h5')
			model_dir_path = os.path.dirname(os.path.realpath(__file__))
			actor_dir_path = model_dir_path + "/agent_weights/jackal/camera/actor_weights_episode" + str(self.load_episode)+ ".h5"
			self.actor.load_weights(actor_dir_path)
    

	def get_action(self, goal, depth_image):
		if goal[0] < 0.15:
			print('Goal reached...')
			return np.array([0.0, 0.0], dtype= np.float32)

		goal = tf.reshape(goal, [1,2])
		depth_image = tf.reshape(depth_image, [1,self.image_height, self.image_width,1])
		pred_action = self.actor(goal, depth_image)
		print("pred_action:", pred_action)
		return tf.reshape(pred_action, [2,])
		#return np.array([0.0, 0.0], dtype= np.float32)
