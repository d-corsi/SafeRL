from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import gym
import random

class DDPG:
	def __init__(self, env, verbose, render=False):
		self.env = env
		self.verbose = verbose
		self.render = render

		self.input_shape = self.env.observation_space.shape
		self.action_shape = env.action_space.shape
		self.action_space = env.action_space.shape[0]

		self.actor = self.get_actor_model(self.input_shape, self.action_space, [env.action_space.low, env.action_space.high])
		self.critic = self.get_critic_model(self.input_shape, self.action_shape)
		self.critic_target = self.get_critic_model(self.input_shape, self.action_shape)
		self.critic_target.set_weights(self.critic.get_weights())

		self.actor_optimizer = keras.optimizers.Adam()
		self.critic_optimizer = keras.optimizers.Adam()
		self.gamma = 0.99
		self.memory_size = 50000
		self.batch_size = 64
		self.exploration_rate = 1.0
		self.exploration_decay = 0.99
		self.tau = 0.005

		self.run_id = np.random.randint(0, 1000)
		
	
	def loop( self, num_episodes=1000 ):
		reward_list = []
		success_list = []
		cost_list = []
		success_list = []
		success_mean_list = deque(maxlen=100)
		replay_buffer = deque(maxlen=self.memory_size)

		for episode in range(num_episodes):
			state = self.env.reset()
			ep_reward = 0
			ep_cost = 0

			while True:
				if self.render: self.env.render()

				action = self.get_action(state)
				new_state, reward, done, info = self.env.step(action)
				ep_reward += reward
				ep_cost += info['state_cost']

				replay_buffer.append([state, action, reward, new_state, done])
				if done: break
				state = new_state

				self.update_networks(replay_buffer)
				self._update_target(self.critic.variables, self.critic_target.variables, tau=self.tau)

			success_mean_list.append( info['goal_reached'] )
			success_list.append( int(np.mean(success_mean_list)*100) )
			reward_list.append( ep_reward )
			cost_list.append( ep_cost )

			self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05	
			
			if self.verbose > 0:
				print(f"(DDPG) Ep: {episode:4}, reward: {ep_reward:6.2f}, success_last_100: {success_list[-1]:3}%, cost_last_100: { np.mean(cost_list[-min(episode, 100):]):5.2f} (eps: {self.exploration_rate:.2f})") 
			if self.verbose > 1: 
				np.savetxt(f"data/success_DDPG_{self.run_id}.txt", success_list)
				np.savetxt(f"data/cost_DDPG_{self.run_id}.txt", cost_list)


	def get_action(self, state):
		action = self.actor(state.reshape((1, -1))).numpy()[0]
		action += np.random.normal(loc=0, scale=self.exploration_rate)

		return action

	
	def _update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))

	
	def update_networks(self, replay_buffer):
		samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)), dtype=object)
		
		with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
			objective_function_c = self.critic_objective_function(samples) #Compute loss with custom loss function
			objective_function_a = self.actor_objective_function(samples) #Compute loss with custom loss function

			grads_c = tape_c.gradient(objective_function_c, self.critic.trainable_variables) #Compute gradients critic for network
			grads_a = tape_a.gradient(objective_function_a, self.actor.trainable_variables) #Compute gradients actor for network

			self.critic_optimizer.apply_gradients( zip(grads_c, self.critic.trainable_variables) ) #Apply gradients to update network weights
			self.actor_optimizer.apply_gradients( zip(grads_a, self.actor.trainable_variables) ) #Apply gradients to update network weights


	def get_actor_model(self, input_shape, output_size, output_range):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(output_size, activation='sigmoid', kernel_initializer=last_init)(hidden_1)

		# Fix output range with the range of the action
		outputs = outputs * (output_range[1] - output_range[0]) + output_range[0]

		return keras.Model(inputs, outputs)


	def get_critic_model(self, input_shape, actor_action_shape):
		inputs = keras.layers.Input(shape=input_shape)
		hidden_inputs = keras.layers.Dense(32, activation='relu')(inputs)

		input_actions = keras.layers.Input(shape=actor_action_shape)
		hidden_input_action = keras.layers.Dense(32, activation='relu')(input_actions)

		concat = keras.layers.Concatenate()([hidden_inputs, hidden_input_action])

		hidden_0 = keras.layers.Dense(64, activation='relu')(concat)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(1, activation='linear')(hidden_1)

		return keras.Model([inputs, input_actions], outputs)


	def actor_objective_function(self, replay_buffer):
		# Extract values from buffer
		state = np.vstack(replay_buffer[:, 0])
		action = np.vstack(replay_buffer[:, 1])

		action = self.actor(state)	
		target = self.critic([state, action])

		return -tf.math.reduce_mean(target)


	def critic_objective_function(self, replay_buffer, gamma=0.99):
		# Extract values from buffer
		state = np.vstack(replay_buffer[:, 0])
		action = np.vstack(replay_buffer[:, 1])
		reward = np.vstack(replay_buffer[:, 2])
		new_state = np.vstack(replay_buffer[:, 3])
		done = np.vstack(replay_buffer[:, 4])

		# Compute the objective function
		# => r + gamma * max(Q(s', a'))
		# L'idea sarebbe quella di utilizzare la Bellman, in particolare vorrei avere il q value dell'azione col valore più alto a pArteire dallo stato s'
		# Trovandomi però con un action sapce continuo non posso testare le diverse azioni
		# Prendo quindi come azione migliore quella che la policy avrebbe scelto in quello stato ,e suppongo sarebbe la scelta migliore
		best_action = self.actor(new_state)
		max_q = self.critic_target([new_state, best_action])

		target = reward + gamma * max_q * (1 - done.astype(int))

		predicted_values = self.critic([state, action])
		mse = tf.math.square(predicted_values - target.numpy())

		return tf.math.reduce_mean(mse)