import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import abc, csv

class ReinforcementLearning( metaclass=abc.ABCMeta ):


	def __init__( self, env, verbose, str_mod, seed ):

			if seed is None: seed = np.random.randint(0, 1000)

			tf.random.set_seed( seed )
			np.random.seed( seed )

			self.env = env
			self.verbose = verbose
			self.run_id = seed
			self.str_mod = str_mod
			self.render = False

			self.input_shape = self.env.observation_space.shape
			self.action_space = env.action_space


	def loop( self, num_episodes=10000 ):		

		# Initialize the loggers
		logger_dict = { "reward": [], "success": [], "step": [], "cost": []  }
		#trajectory_dict = { "state": [], "action": [] }

		# Setup the environment for for the I/O on file (logger/models)
		# (only when verbose is active on file)
		if self.verbose > 1:

			# Create the string with the configuration for the file name
			file_name = f"{self.str_mod}_seed{self.run_id}"

			#
			for key, value in self.relevant_params.items():
				file_name += f"_{value}{self.__dict__[key]}"

			# Create the CSV file and the writer
			csv_file = open( f"data/{file_name}.csv", mode='w')
			fieldnames = ['episode', 'reward', 'success', 'step', 'cost']
			self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
			self.writer.writeheader()

			# Support variable to limit the number of msaved models
			last_saved = 0

		# Iterate the training loop over multiple episodes
		for episode in range(num_episodes):

			# Reset the environment at each new episode
			state = self.env.reset()

			# Initialize the values for the logger
			logger_dict['reward'].append(0)
			logger_dict['success'].append(0)
			logger_dict['step'].append(0)
			logger_dict['cost'].append(0)
			#trajectory_dict['state'].append([])
			#trajectory_dict['action'].append([])

			# Main loop of the current episode
			while True:

				#
				if self.render: self.env.render()
				
				# Select the action, perform the action and save the returns in the memory buffer
				action, action_prob = self.get_action(state)
				new_state, reward, done, info = self.env.step(action)
				self.memory_buffer.append([state, action, action_prob, reward, new_state, done, info['state_cost']])
				
				# Update the dictionaries for the logger and the trajectory
				logger_dict['reward'][-1] += reward	
				logger_dict['step'][-1] += 1	
				logger_dict['cost'][-1] += info['state_cost']
				logger_dict['success'][-1] = 1 if info['goal_reached'] else 0
				#trajectory_dict['state'][-1].append( state )
				#trajectory_dict['action'][-1].append( state )	
				
				# Call the update rule of the algorithm
				self.network_update_rule( episode, done )

				# Exit if terminal state and eventually update the state
				if done: break
				state = new_state

			# Log all the results, depending on the <verbose> parameter
			# here simple print of the results
			if self.verbose > 0:
				last_n =  min(len(logger_dict['reward']), 100)
				reward_last_100 = logger_dict['reward'][-last_n:]
				cost_last_100 = logger_dict['cost'][-last_n:]
				step_last_100 = logger_dict['step'][-last_n:]
				success_last_100 = logger_dict['success'][-last_n:]

				print( f"({self.str_mod}) Ep: {episode:5}", end=" " )
				print( f"reward: {logger_dict['reward'][-1]:5.2f} (last_100: {np.mean(reward_last_100):5.2f})", end=" " )
				print( f"cost_last_100: {int(np.mean(cost_last_100)):3d}", end=" " )
				print( f"step_last_100 {int(np.mean(step_last_100)):3d}", end=" " )
				print( f"success_last_100 {int(np.mean(success_last_100)*100):4d}%" )

			# Log all the results, depending on the <verbose> parameter
			# here save the log to a CSV file
			if self.verbose > 1:	
				self.writer.writerow({ 
					'episode' : episode,
					'reward': logger_dict['reward'][-1], 
					'success': logger_dict['success'][-1], 
					'step': logger_dict['step'][-1], 
					'cost': logger_dict['cost'][-1]			
				})

			# Log all the results, depending on the <verbose> parameter
			# here save the models generated models if "good"
			success_rate = int(np.mean(success_last_100) * 100)
			if self.verbose > 2 and success_rate > 98 and episode > 100 and (episode - last_saved) > 30: 
				self.actor.save(f"models/{self.str_mod}_id{self.run_id}_ep{episode}_{self.nodes}.h5")
				last_saved = episode


	def generate_model( self, input_shape, output_size=1, layers=2, nodes=32, last_activation='linear' ):

		#
		hiddens_layers = [tf.keras.layers.Input( shape=input_shape )]
		for _ in range(layers):	hiddens_layers.append( tf.keras.layers.Dense( nodes, activation='relu')( hiddens_layers[-1] ) )
		hiddens_layers.append( tf.keras.layers.Dense( output_size, activation=last_activation)( hiddens_layers[-1] ) )	

		#
		return tf.keras.Model( hiddens_layers[0], hiddens_layers[-1] )