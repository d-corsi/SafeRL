from algos.custom_gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class PPO( ReinforcementLearning ):

	def __init__( self, env, verbose, str_mod="PPO", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 10
		self.critic_epoch = 40
		self.critic_batch_size = 128
		self.trajectory_mean = False
		self.layers = 2
		self.nodes = 32

		# 
		self.relevant_params = {
			'layers' : 'lay',
			'nodes' : 'nod',
			'gamma' : 'gamma',
			'trajectory_update' : 'tu',
			'critic_epoch' : 'ce',
			'critic_batch_size' : 'cbs'
		}

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		#
		self.memory_buffer = deque( maxlen=self.memory_size )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space.n, \
			layers=self.layers, nodes=self.nodes, last_activation='softmax')
		self.critic = self.generate_model(self.input_shape)

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()


	def network_update_rule( self, episode, terminal ):

		# Update of the networks for PPO!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	def update_networks( self, memory_buffer ):

		# Critic update (repeated epoch times on a batch, fixed):
		for _ in range( self.critic_epoch ):
			idx = np.random.randint(memory_buffer.shape[0], size=self.critic_batch_size)
			training_batch = memory_buffer[idx]
			with tf.GradientTape() as critic_tape:
				critic_objective_function = self.critic_objective_function( training_batch )
				critic_gradient = critic_tape.gradient( critic_objective_function, self.critic.trainable_variables )
				self.critic_optimizer.apply_gradients( zip(critic_gradient, self.critic.trainable_variables) )

		# Actor update (repeated #epoch times):
		with tf.GradientTape() as actor_tape:
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	def temporal_difference(self, reward, new_state, done): 
		return reward + (1 - done.astype(int)) * self.gamma * self.critic(new_state) 

	
	def critic_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		predicted_value = self.critic(state)
		target = self.temporal_difference(reward, new_state, done)
		mse = tf.math.square(predicted_value - target)

		return tf.math.reduce_mean(mse)


	def get_action(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space.n, p=softmax_out.numpy()[0])
		return selected_action, softmax_out[0][selected_action]


	def actor_objective_function( self, memory_buffer ):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# Computation of the advantege
		baseline = self.critic(state)
		advantage = self.temporal_difference(reward, new_state, done) - baseline

		prob = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)] #Trick to obatin the coordinates of each desired action
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		clip_val = 0.2
		obj_1 = r_theta * advantage
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * advantage
		partial_objective = tf.math.minimum(obj_1, obj_2)

		# Mean over the trajectory (OpeanAI implementation)
		if self.trajectory_mean: -tf.math.reduce_mean(partial_objective)

		# Sum over the trajectory (Theoretical implementation)
		return -tf.math.reduce_sum(partial_objective)
	