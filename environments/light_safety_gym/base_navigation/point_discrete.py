from light_safety_gym.base_navigation.base_navigation import BaseNavigation
import gym
import numpy as np

class PointNavigationDiscrete( BaseNavigation ):


	def __init__(self, **kwargs):

		# Load from the super class
		super().__init__()

		# Parse kwargs attribute to modify the standard settings
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Definition of the discrete gym action space
		self.action_space = gym.spaces.Discrete( len(self.actions) )

	
	def perform_action(self, action):
		
		# Perform the actions
		self.agent_angle += self.actions[action][0]

		# Compute the mathematical model
		delta_x = np.math.cos( self.agent_angle-np.pi/2 ) * self.actions[action][1]
		delta_y = np.math.sin( self.agent_angle-np.pi/2 ) * self.actions[action][1]
		self.agent_position[0] -= delta_x
		self.agent_position[1] -= delta_y
	