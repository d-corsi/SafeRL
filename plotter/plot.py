from MyPlotter import MyPlotter
import glob


# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="success (%)", title="Success Rate" )
plotter.load_array([
		glob.glob("data/success_IPO_*.txt"),
		glob.glob("data/success_PPO_*.txt")
])
plotter.process_data( rolling_window=500, starting_pointer=100 )
plotter.render_std_log( labels=["IPO", "PPO"], colors=["r", "g"] )

# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="cost", title="Episode Cost", cap=60  )
plotter.load_array([
		glob.glob("data/cost_IPO_*.txt"),
		glob.glob("data/cost_PPO_*.txt")
])
plotter.process_data( rolling_window=500, starting_pointer=100 )
plotter.render_std_log( labels=["IPO", "PPO"], colors=["r", "g"] )