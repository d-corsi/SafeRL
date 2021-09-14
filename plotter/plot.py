from MyPlotter import MyPlotter
import glob

# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="success (%)", title="Success Rate" )
plotter.load_array([
		glob.glob("data/success_DDPG_*.txt"),
		glob.glob("data/success_PPO_*.txt")
])
plotter.process_data( rolling_window=1, starting_pointer=0 )
plotter.render( labels=["DDPG", "PPO"], colors=["r", "g"] )

# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="cost", title="Episode Cost", cap=30  )
plotter.load_array([
		glob.glob("data/cost_DDPG_*.txt"),
		glob.glob("data/cost_PPO_*.txt")
])
plotter.process_data( rolling_window=100, starting_pointer=0 )
plotter.render( labels=["DDPG", "PPO"], colors=["r", "g"] )