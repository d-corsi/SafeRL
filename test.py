# Suppress TF2 warnings and load libraries
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym, sys


# Load the PPO algorithm for discrete action space and start the training
def main_discrete_PPO():
	from algorithms.PPO import PPO

	env = gym.make( "light_safety_gym:point_discrete-v0" )
	learner = PPO( env=env, verbose=2, render=False )
	learner.loop( 5000 )


# Load the IPO algorithm for discrete action space and start the training
def main_discrete_IPO():
	from algorithms.IPO import IPO

	env = gym.make( "light_safety_gym:point_discrete-v0" )
	learner = IPO( env=env, verbose=2, cost_limit=20, render=False )
	learner.loop( 5000 )


# Load the DDPG algorithm for continuous action space and start the training
def main_continuous_DDPG():
	from algorithms.DDPG import DDPG
	env = gym.make( "light_safety_gym:point_continuous-v0" )
	learner = DDPG( env=env, verbose=2, render=False )
	learner.loop( 5000 )


# Parse the input command and run the correct function
if __name__ == "__main__":
	if (sys.argv[1] == "-PPO_discrete"): main_discrete_PPO()
	elif (sys.argv[1] == "-IPO_discrete"): main_discrete_IPO()
	elif (sys.argv[1] == "-DDPG_continuous"): main_continuous_DDPG()
	else: raise ValueError(f"Invalid command: '{sys.argv[1]}' (options: [-PPO_discrete, -IPO_discrete, -DDPG_continuous])")
