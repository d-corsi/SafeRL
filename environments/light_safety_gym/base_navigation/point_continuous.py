from light_safety_gym.base_navigation.base_navigation import BaseNavigation
import gym
import numpy as np

class PointNavigationContinuous( BaseNavigation ):


	def __init__(self, **kwargs):

		# Load from the super class
		super().__init__()

		# Parse kwargs attribute to modify the standard settings
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Definition of the continuous gym action space
		self.action_space = gym.spaces.Box( np.array([-self.angular_velocity, 0]), np.array([self.angular_velocity, self.max_linear_velocity]) )


	def perform_action(self, action):
		# Perform the actions
		self.agent_angle += action[0]

		# Compute the mathematical model
		delta_x = np.math.cos( self.agent_angle-np.pi/2 ) * action[1]
		delta_y = np.math.sin( self.agent_angle-np.pi/2 ) * action[1]
		self.agent_position[0] -= delta_x
		self.agent_position[1] -= delta_y
	