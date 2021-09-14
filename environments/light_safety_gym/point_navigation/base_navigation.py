import gym, abc
import numpy as np

# This class implements abstract methods
class BaseNavigation( gym.Env, metaclass=abc.ABCMeta ):

	def __init__(self):

		# Constant definition
		self.world_size = 800
		self.spawn_size = 700

		self.agent_size = 5
		self.goal_size = 10

		self.max_linear_velocity = 5
		self.min_linear_velocity = 2
		self.angular_velocity = 0.3

		self.lidar_sphere = 3
		self.lidar_length = 15 
		self.lidar_density = 12
		self.render_lidar = False

		self.obstacle_number = 15
		self.obstacle_size = [20, 30]
		self.obstacle_list = []

		# Variables definition
		self.max_step = 400
		self.episode_step = 0
		self.old_distance = 0

		# Definition of the possibile discrete actions
		self.actions = [
			(0, self.max_linear_velocity),	
			(self.angular_velocity, 0),
			(-self.angular_velocity, 0),
			(self.angular_velocity, self.min_linear_velocity),
			(-self.angular_velocity, self.min_linear_velocity)			
		]

		# Definition of the discrete gym action space
		self.action_space = gym.spaces.Discrete(len(self.actions))

		# Definition of the continuous gym observation space
		self.observation_space = gym.spaces.Box(
			np.array([-1, 0] + [0 for _ in range(self.lidar_density)]),
			np.array([ 1, 1] + [1 for _ in range(self.lidar_density)]),
			dtype=np.float32
		)

		# Internal state of the agent
		self.agent_position = [0, 0]
		self.agent_angle = 0
		self.goal_position = [0, 0]
		self.lidar_matrix = np.array([])

		# Render attributes
		self._viewer = None
		self._viewer_objs_holder = {}

	
	def reset(self):

		# Randomize the starting position and rotation of the agent
		self.agent_angle = np.random.random() * 2 * np.pi - np.pi
		self.agent_position = [
			np.random.randint(self.world_size-self.spawn_size, self.spawn_size), 
			np.random.randint(self.world_size-self.spawn_size, self.spawn_size)
			]

		# Randomize the position of the goal
		self.goal_position = [
			np.random.randint(self.world_size-self.spawn_size, self.spawn_size), 
			np.random.randint(self.world_size-self.spawn_size, self.spawn_size)
			]

		# Randomize the obstacles, initialize the array with empty values
		self.obstacle_list = [(0, 0, 0) for _ in range(self.obstacle_number)]
		# Iterate on the obstacles list to generate the random obstacles
		for k in range(self.obstacle_number):
			# Try to generate the obstacle until it does not overlaps with other objects
			while True:
				# Random generate position and size of the obstacle
				self.obstacle_list[k] = (np.random.randint(self.world_size-self.spawn_size, self.spawn_size), 
										np.random.randint(self.world_size-self.spawn_size, self.spawn_size), 
										np.random.randint(self.obstacle_size[0], self.obstacle_size[1])
										) 
				# Suppose no overlap between the obstacle and any other item
				overlap_flag = False
				# Iterate on the already created obstacles, goal and agent
				# to check the collision
				for item in self.obstacle_list[:k] + [(self.goal_position[0], self.goal_position[1], self.goal_size), 
													  (self.agent_position[0], self.agent_position[1], self.agent_size)]:
					# If the obstacle overlpas set the overlap flag to true
					if self.object_overlap( (self.obstacle_list[k][0], self.obstacle_list[k][1]), self.obstacle_list[k][2],
											(item[0], item[1]), item[2]					
										   ):
						overlap_flag = True
				# If the overlap flag is false (i.e., no overlaps) break the while true cycle and pass to the next obstacle
				if not overlap_flag: break
			

		# Reset variables
		self.episode_step = 0
		self.old_distance = self.get_distance(self.agent_position, self.goal_position)

		# Update the lidar values
		self.lidar_matrix = self.update_lidar_matrix()

		# Call the state normalizer function
		return self.get_state()


	def step(self, action):

		# Initialize the return objects
		reward = -0.005
		done = False

		# Initialize info dictionary
		info = {}
		info['goal_reached'] = 0
		info['state_cost'] = 0

		# Update variables
		self.episode_step += 1

		# Update lidar info
		self.lidar_matrix = self.update_lidar_matrix()

		# Call to update position and rotation of the agent
		self.perform_action( action )

		# Compute reward function and update the old distance for the reward computation
		reward += 0.01 if (self.old_distance > self.get_distance(self.agent_position, self.goal_position)) else -0.01
		self.old_distance = self.get_distance(self.agent_position, self.goal_position)

		# Check for goal collision
		for obstacle_def in self.obstacle_list:
			if self.agent_overlap([obstacle_def[0], obstacle_def[1]], obstacle_def[2]):
				info['state_cost'] = 1

		# Check for goal reached
		if self.agent_overlap(self.goal_position, self.goal_size):
			done = True
			info['goal_reached'] = 1
			reward = 1

		# Check for the max step
		if self.episode_step > self.max_step:
			done = True

		# Call the state normalizer function and other gym requirements
		return self.get_state(), reward, done, info


	def close(self):
		if self._viewer:
			self._viewer.close()
			self._viewer = None


	############################
	## Private Suppor Methods ##
	############################


	# Check if the agent collides with another object
	def agent_overlap(self, target_position, target_radius):
		# Call the general overlap checker method with the specific of the agent
		return self.object_overlap(self.agent_position, self.agent_size, target_position, target_radius)


	# Check if an object overlaps another object (only circle)
	def object_overlap(self, object_position, object_radius, target_position, target_radius):
		# Compute the distance between the center og the two objects
		distance = self.get_distance(object_position, target_position)
		# Check if the distance is less than the sum of the two radius
		return distance < (object_radius + target_radius)


	# Standard method to compute the distance between two poin in cartesian space
	def get_distance(self, object_position, target_position):
		delta_x = (object_position[0] - target_position[0])**2
		delta_y = (object_position[1] - target_position[1])**2
		return np.math.sqrt( delta_x + delta_y )
		

	# Method to compute the current normalized observation space 
	def get_state(self):

		# Update the lidar values and normalize in [0, 1]
		lidar_state = self.update_lidar_state() 
		normalized_lidar_state = lidar_state / [self.lidar_length for _ in range(self.lidar_density)]

		# Compute the compass respect the north, the heading respect the goal and nomrlize in [0, 360]
		compass = (np.degrees(self.agent_angle) + 90)
		heading = np.degrees( np.math.atan2(self.goal_position[1]-self.agent_position[1], self.goal_position[0]-self.agent_position[0]) )
		heading_fixed = (max(compass, heading) - min(compass, heading)) % 360

		# Normalize the heading in [-180, 180]
		heading_fixed = (heading_fixed * -1) if heading_fixed < 180 else (360 - heading_fixed)
		
		# Compute the distance and the diagonal of the spawn space as normalization factor
		distance = self.get_distance(self.agent_position, self.goal_position)
		normalize_factor = self.get_distance([0, 0], [self.spawn_size, self.spawn_size])

		# Normalize heading in [-1, 1] and distance in [0, 1]
		state = np.array([heading_fixed, distance])
		normalized_state = state / [180.0, normalize_factor] 

		# Return the heading (normalized [-1, 1]), distance (normalized [0, 1]) and lidar state (normalized [0, 1])
		return np.concatenate((normalized_state, normalized_lidar_state))


	# Method to update the lidar matrix, compute the points of the sphere in the current time step
	def update_lidar_matrix(self):

		#  Initialize the lidar matrix, the structure is a matrix of a row for each lidar line with the
		# list of the sphere. Fix of 90 deg to start in front of the agent
		lidar_total_list = []		
		compass = np.degrees(self.agent_angle) + 90

		# Compute the position of the sphere for each scan and add to the matrix
		for angle_step in [(360 / self.lidar_density * i) for i in range(self.lidar_density)]:
			angle = np.radians((compass + angle_step) % 360)
			for r in [self.lidar_sphere*2*i for i in range(self.lidar_length)]:
				delta_x = np.math.cos( angle ) * r
				delta_y = np.math.sin( angle ) * r		
				new_point = [self.agent_position[0] + delta_x, self.agent_position[1] + delta_y]
				lidar_total_list.append( new_point )
			
		# Return the lidar list in the correct shape described before
		return np.array(lidar_total_list).reshape(self.lidar_density, -1, 2)


	# Compute the value of the lidar sensor for each direction, computing the distance with the obstacles
	def update_lidar_state(self):

		# State contain the index first sphere that coliides for each lidar line
		lidar_state = []

		# Iterate for each lidar line, each possibile obsacle and each sphere
		for lidar_line in self.lidar_matrix:
			first_found = self.lidar_matrix.shape[1]
			for idx, sphere in enumerate(lidar_line):
				for obstacle_def in self.obstacle_list:

					# Check the collision between one of the sphere of the scan and one obstacle
					if self.object_overlap( (sphere[0], sphere[1]), self.lidar_sphere, (obstacle_def[0], obstacle_def[1]), obstacle_def[2] ):

						# Always save the closest obsatcle of the line respect the agent
						first_found = min(first_found, idx)

			# Update the matrix
			lidar_state.append( first_found )

		# Return the lidar staet, values represent the distance of the obstacle from the agent
		# for each direction, in number of sphere of the lidar line
		return np.array(lidar_state)


	# Abstract method that perform the action on the mathematical model
	# to implement in the child class
	@abc.abstractmethod
	def perform_action(self, action):
		return


	############################
	#### Gym Render Method #####
	############################


	def render(self, mode='human'):
		from gym.envs.classic_control import rendering

		# Color definitions
		agent_color = (.0, .0, .0)
		goal_color = (.99, .99, .0)
		bakground_color_0 = (.0, .8, .0)
		bakground_color_1 = (.0, .5, .0)
		lidar_color = (.99, .0, .0)
		obstacle_color = (.99, .0, .0, .7)

		# Create the viewer and intialize objects
		if self._viewer is None:			

			# Initialize the new viewer
			self._viewer = rendering.Viewer( self.world_size , self.world_size )

			# Draw Background 0 (Static)
			view_background = rendering.make_polygon( [[0, 0], [self.world_size, 0], [self.world_size, self.world_size], [0, self.world_size]] )
			view_background.set_color( *bakground_color_0 )
			self._viewer.add_geom(view_background)

			# Draw Background 1 (Static)
			delta_size = self.world_size-self.spawn_size
			view_background = rendering.make_polygon([
				[self.spawn_size, self.spawn_size], 
				[delta_size, self.spawn_size], 
				[delta_size, delta_size], 
				[self.spawn_size, delta_size]
			])
			view_background.set_color( *bakground_color_1 )
			self._viewer.add_geom(view_background)
		
			# Draw Goal Objects (Static)
			view_goal = rendering.make_circle( self.goal_size )
			view_goal.set_color( *goal_color )
			self._viewer_objs_holder["goal_trans"] = rendering.Transform()
			view_goal.add_attr(self._viewer_objs_holder["goal_trans"])
			self._viewer.add_geom(view_goal)

			# Draw Agent Object (Dynamic)
			view_agent = rendering.make_circle( self.agent_size )
			view_agent.set_color( *agent_color )
			self._viewer_objs_holder["agent_trans"] = rendering.Transform()
			view_agent.add_attr(self._viewer_objs_holder["agent_trans"])
			self._viewer.add_geom(view_agent)

			# Draw Agent Directio Marker
			view_direction = rendering.make_circle(self.agent_size / 3)
			view_direction.set_color( *agent_color )
			self._viewer_objs_holder["direction_trans"] = rendering.Transform( translation=(0, self.agent_size) )
			view_direction.add_attr(self._viewer_objs_holder["direction_trans"])
			view_direction.add_attr(self._viewer_objs_holder["agent_trans"])
			self._viewer.add_geom(view_direction)

		# Manage goal movement
		self._viewer_objs_holder["goal_trans"].set_translation(*self.goal_position)
		
		# Manage agent movement
		self._viewer_objs_holder["agent_trans"].set_translation(*self.agent_position)
		self._viewer_objs_holder["agent_trans"].set_rotation(self.agent_angle)

		# Render obstacles
		for obstacle_def in self.obstacle_list:
			t = rendering.Transform( translation=(obstacle_def[0], obstacle_def[1]) )
			obj = self._viewer.draw_circle( obstacle_def[2] )
			obj.add_attr(t)
			obj._color.vec4 = obstacle_color

		# Render lidar scans
		if self.render_lidar:
			for render_point in self.lidar_matrix.reshape(-1, 2):
				t = rendering.Transform( translation=(render_point[0], render_point[1]) )
				self._viewer.draw_circle( self.lidar_sphere, color=lidar_color).add_attr(t)

		# Return gym render
		return self._viewer.render( return_rgb_array=(mode=='rgb_array') )
