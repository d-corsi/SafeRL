# Suppress TF2 warnings and load libraries
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym, sys


# Load the PPO algorithm for discrete action space and start the training
def main_discrete():
	from algorithms.PPO import PPO
	env = gym.make( "light_safety_gym:discrete_navigation-v0" )
	learner = PPO( env=env, verbose=2, render=False )
	learner.loop( 1000 )


# Load the DDPG algorithm for continuous action space and start the training
def main_continuous():
	from algorithms.DDPG import DDPG
	env = gym.make( "light_safety_gym:continuous_navigation-v0" )
	learner = DDPG( env=env, verbose=2, render=False )
	learner.loop( 1000 )


# Parse the input command and run the correct function
if __name__ == "__main__":
	if (sys.argv[1] == "-discrete"): main_discrete()
	elif (sys.argv[1] == "-continuous"): main_continuous()
	else: raise ValueError(f"Invalid command: '{sys.argv[1]}' (options: [-discrete, -continuous])")
