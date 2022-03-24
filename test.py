# Suppress TF2 warnings and load libraries
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym, sys


# Load the PPO algorithm for discrete action space and start the training
def main_discrete_PPO( env, verbose=1, episodes=5000, render=False ):
	from algos.PPO import PPO

	learner = PPO( env=env, verbose=verbose, render=render )
	learner.loop( episodes )


# Load the IPO algorithm for discrete action space and start the training
def main_discrete_IPO( env, verbose=1, episodes=5000, render=False ):
	from algos.IPO import IPO

	learner = IPO( env=env, verbose=verbose, cost_limit=15, render=render )
	learner.loop( episodes )


# Load the Lagrangian PPO algorithm for discrete action space and start the training
def main_discrete_Lagrangian( env, verbose=1, episodes=5000, render=False ):
	from algos.Lagrangian import Lagrangian

	learner = Lagrangian( env=env, verbose=verbose, cost_limit=15, render=render )
	learner.loop( episodes )


# Parse the input command and run the correct function
if __name__ == "__main__":
	env = gym.make( "light_safety_gym:point_discrete-v0" )
	main_discrete_PPO( env, verbose=1, episodes=5000 )
	main_discrete_Lagrangian( env, verbose=1, episodes=5000 )
	quit()

	if (sys.argv[1] == "-PPO"): main_discrete_PPO()
	elif (sys.argv[1] == "-IPO"): main_discrete_IPO()
	elif (sys.argv[1] == "-Lagr"): main_discrete_Lagrangian()
	else: raise ValueError(f"Invalid command: '{sys.argv[1]}' (options: [-PPO, -IPO, -Lagr])")
