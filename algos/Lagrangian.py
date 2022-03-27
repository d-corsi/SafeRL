from algos.custom_gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class Lagrangian( ReinforcementLearning ):

	"""
	Class that inherits from ReinforcementLearning to implements the REINFORCE algorithm, the original paper can be found here:
	https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf [1]

	[1] Policy Gradient Methods for Reinforcement Learning with Function Approximation, 
		Sutton et al., 
		Advances in neural information processing systems, 1999

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod="Lagrangian", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space.n, last_activation='softmax')

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()

		#
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 5
		self.trajectory_mean = False
		self.lagrangian_var = 1
		self.cost_limit = 50

		# 
		self.relevant_params = {
			'gamma' : 'gamma',
			'trajectory_update' : 'tu',
			'lagrangian_var' : 'lambda',
			'cost_limit' : 'climit'
		}

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		self.memory_buffer = deque( maxlen=self.memory_size )
		


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
	def network_update_rule( self, episode, terminal ):

		# Update of the networks for Reinforce!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

		
		# Extract values from buffer for the advantage computation
		cost = memory_buffer[:, 6]
		done = np.vstack(memory_buffer[:, 5])

		end_trajectories = np.where(done == True)[0]

		#
		trajectory_cost = []
		counter = 0
		for i in end_trajectories:
			trajectory_cost.append( sum(cost[counter : i+1]) )
			counter = i+1

		# Lagrangian variable update (simulation of 1-variable gradient step)
		# simulation of a SGD with a fixed learning rate of 0.05
		cost_barrier = np.mean(trajectory_cost) - self.cost_limit
		if cost_barrier <= 0: self.lagrangian_var -= 0.05
		else: self.lagrangian_var += 0.05
		
		# Limit of the lagrangian multiplier >= 0
		if self.lagrangian_var < 0: self.lagrangian_var = 0
		
		# Actor update (repeated 1 time for each call):
		with tf.GradientTape() as actor_tape:
			
			# Compute the objective function, compute the gradient information and apply the
			# gradient with the optimizer
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	# Mandatory method to implement for the ReinforcementLearning class
	# here we select thea action based on the state, for policy gradient method we obtain
	# a probability from the network, from where we perform a sampling
	def get_action(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space.n, p=softmax_out.numpy()[0])
		return selected_action, None

	
	# Computing the objective function of the actor for the gradient ascent procedure,
	# here is where the 'magic happens'
	def actor_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = memory_buffer[:, 3]
		action = memory_buffer[:, 1]
		done = np.vstack(memory_buffer[:, 5])
		cost = memory_buffer[:, 6]

		# For multiple trajectories find the end of each one
		end_trajectories = np.where(done == True)[0]
		
		# Extract the proability of the action with the crurent policy,
		# execute the current policy on the state to obtain the probailities for each action
		# than, using the action selected by the network in the buffer, compute the probability 
		# from the output. Notice that i need to execute again the network and can not use the probaiblity in the buffer
		# computed at runtime beacuse we need the gradient informations
		probability = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)

		# Computation of the log_prob and the sum of the reward for each trajectory.
		# To obtain the probability of the trajectory i need to sum up the values for each single trajectory and multiply 
		# this value for the cumulative reward (no discounted or 'reward to go' for this vanilla implementation).
		trajectory_probabilities = []
		trajectory_rewards = []
		trajectory_cost = []
		counter = 0
		for i in end_trajectories:
			trajectory_probabilities.append( tf.math.reduce_sum( tf.math.log(probability[counter : i+1])) )
			trajectory_rewards.append( sum(reward[counter : i+1]) )
			trajectory_cost.append( sum(cost[counter : i+1]) )
			counter = i+1

		# Multiplication of log_prob times the reward of the trajectory
		# here we obtain an array of N elements where N is the number of trajectories (this
		# value depends on the parameter <trajectory_update>).
		trajectory_objectives = []
		for log_prob, rw, cs in zip(trajectory_probabilities, trajectory_rewards, trajectory_cost):
			#trajectory_objectives.append( log_prob * (rw - np.mean(trajectory_rewards)) )
			trajectory_objectives.append( log_prob * (rw - self.lagrangian_var * (cs - self.cost_limit)) )

		# Computing the mean value between all the trajectories, this introduce siamo variance but reduce 
		# the bias, see the original paper for more details about the baseline
		objective_function = tf.reduce_mean( trajectory_objectives )

		# NB: returna negative value to automatically use a gradient ascent approach
		# on TensorFlow
		return -objective_function
	