from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import random

class IPO:
	def __init__(self, env, verbose, cost_limit, render=False):
		self.env = env
		self.verbose = verbose
		self.render = render

		self.input_shape = self.env.observation_space.shape
		self.action_space = env.action_space.n

		self.actor = self.get_actor_model_disc(self.input_shape, self.action_space)
		self.get_action = self.get_action_disc
		self.actor_objective_function = self.actor_objective_function_disc

		self.critic = self.get_critic_model(self.input_shape)

		self.actor_optimizer = keras.optimizers.Adam()
		self.critic_optimizer = keras.optimizers.Adam()
		self.gamma = 0.99
		self.batch_size = 500
		self.cost_limit = cost_limit

		self.run_id = np.random.randint(0, 1000)
		

	def loop( self, num_episodes=1000 ):
		reward_list = []
		success_list = []
		cost_list = []
		success_list = []
		success_mean_list = deque(maxlen=100)
		memory_buffer = deque()

		for episode in range(num_episodes):
			state = self.env.reset()
			ep_reward = 0
			ep_cost = 0
			
			while True:
				if self.render: self.env.render()

				action, action_prob = self.get_action(state)
				new_state, reward, done, info = self.env.step(action)
				ep_reward += reward		
				ep_cost += info['state_cost']

				memory_buffer.append([state, action, action_prob, reward, new_state, done, info['state_cost']])
				if done: break
				state = new_state

			success_mean_list.append( info['goal_reached'] )
			success_list.append( int(np.mean(success_mean_list)*100) )
			reward_list.append( ep_reward )
			cost_list.append( ep_cost )

			self.update_networks(np.array(memory_buffer, dtype=object))
			memory_buffer.clear()

			if self.verbose > 0:
				cost_last_100 = np.mean(cost_list[-min(episode, 100):])
				print(f"(IPO) Ep: {episode:4}, reward: {ep_reward:6.2f}, cost: {ep_cost:3d}, success_last_100: {success_list[-1]:3}%, cost_last_100: { cost_last_100:5.2f}") 
			if self.verbose > 1: 
				np.savetxt(f"data/success_IPO_{self.run_id}.txt", success_list)
				np.savetxt(f"data/cost_IPO_{self.run_id}.txt", cost_list)


	def update_networks(self, memory_buffer):
		# Fix cumulative reward on multiple episodes
		memory_buffer[:, 3] = [ el for el in self.discount_reward(memory_buffer[:, 3]) ]
		memory_buffer[:, 6] = [ el for el in self.cumulative_cost_diff(memory_buffer[:, 6]) ]


		with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
			objective_function_c = self.critic_objective_function(memory_buffer) #Compute loss with custom loss function
			objective_function_a = self.actor_objective_function(memory_buffer) #Compute loss with custom loss function

			grads_c = tape_c.gradient(objective_function_c, self.critic.trainable_variables) #Compute gradients critic for network
			grads_a = tape_a.gradient(objective_function_a, self.actor.trainable_variables) #Compute gradients actor for network

			self.critic_optimizer.apply_gradients( zip(grads_c, self.critic.trainable_variables) ) #Apply gradients to update network weights
			self.actor_optimizer.apply_gradients( zip(grads_a, self.actor.trainable_variables) ) #Apply gradients to update network weights


	def discount_reward(self, rewards):
		sum_reward = 0
		discounted_rewards = []

		for r in rewards[::-1]:
			sum_reward = r + self.gamma * sum_reward
			discounted_rewards.append(sum_reward)
		discounted_rewards.reverse() 

		# Normalize
		eps = np.finfo(np.float64).eps.item()  # Smallest number such that 1.0 + eps != 1.0 
		discounted_rewards = np.array(discounted_rewards)
		discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

		return discounted_rewards

	
	def cumulative_cost_diff(self, costs):
		cost_diff = sum(costs) - self.cost_limit
		trajectory_cost_diff = np.array([ cost_diff for _ in costs ])
		return trajectory_cost_diff


	##########################
	##### CRITIC METHODS #####
	##########################


	def get_critic_model(self, input_shape):
		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(1, activation='linear')(hidden_1)

		return keras.Model(inputs, outputs)

	
	def critic_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		discounted_reward = np.vstack(memory_buffer[:, 3])

		predicted_value = self.critic(state)
		target = discounted_reward
		mse = tf.math.square(predicted_value - target)

		return tf.math.reduce_mean(mse)

	
	##########################
	#### DISCRETE METHODS ####
	##########################


	def get_action_disc(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space, p=softmax_out.numpy()[0])
		return selected_action, softmax_out[0][selected_action]


	def actor_objective_function_disc(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		discounted_reward = np.vstack(memory_buffer[:, 3])
		trajectory_cost = np.vstack(memory_buffer[:, 6])

		baseline = self.critic(state)
		adv = discounted_reward - baseline # Advantage = MC - baseline

		prob = self.actor(state)

		action_idx = [[counter, val] for counter, val in enumerate(action)] #Trick to obatin the coordinates of each desired action
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		clip_val = 0.2
		obj_1 = r_theta * adv
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * adv
		partial_objective_reward = tf.math.minimum(obj_1, obj_2)
		
		eps = np.finfo(np.float64).eps.item()
		trajectory_cost_diff = max( -np.mean(trajectory_cost), eps )
		barrier_function = tf.math.log(prob) * trajectory_cost_diff / 40
		
		ipo_objective = tf.add(partial_objective_reward, barrier_function)

		return -tf.math.reduce_mean(ipo_objective)
	
		
	def get_actor_model_disc(self, input_shape, output_size):
		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(output_size, activation='softmax')(hidden_1)

		return keras.Model(inputs, outputs)
