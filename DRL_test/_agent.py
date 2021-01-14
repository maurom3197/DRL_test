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

from NeuralNetworks import  ActorNetwork


def _init_agent(self):
	self.agent = DDPGLidarAgent(
			state_size = 38, 
			action_size = 2, 
			max_linear_vel = 0.8,
			max_angular_vel = 2,
			load = True
			)


class DDPGLidarAgent:

	def __init__(self, state_size, action_size = 2, max_linear_vel = 0.8, max_angular_vel = 2, max_memory_size = 100000, load = False,gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05, tau = 0.01, batch_size = 64, noise_std_dev = 0.2):


		# State size and action size
		self.state_size = state_size 
		self.action_size = action_size 
		self.max_linear_vel = max_linear_vel
		self.max_angular_vel = max_angular_vel

		self.actor = ActorNetwork(self.state_size, self.max_linear_vel, self.max_angular_vel, lr = 0.0001, name = 'actor')


		#Load Models
		self.load = load
		self.load_episode = 0

		if self.load:
			#actor_dir_path = os.path.join(
			#	self.actor.model_dir_path,
			#	'actor_stage1_episode'+str(self.load_episode)+'.h5')
			model_dir_path = os.path.dirname(os.path.realpath(__file__))
			actor_dir_path = model_dir_path + "/actor" + ".h5"
			self.actor.load_weights(actor_dir_path)
    

	def get_action(self, state):
		pred_action = self.actor(state.reshape(1, len(state)))
		pred_action = tf.reshape(pred_action, [2,])
		return pred_action

