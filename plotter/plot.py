from reinforcement_plotter import ReinforcementPlotter
import glob

data = [
		glob.glob("data/PPO_32*.csv"),
		glob.glob("data/PPO_16*.csv"),
		glob.glob("data/PPO_8*.csv")
]

data = [ glob.glob("data/PPO*nod16*.csv") ]

# Plot The Results
plotter = ReinforcementPlotter( x_label="episode", y_label="seccess", title="Network Size Comparison" )
plotter.load_array( data, key="success", ref_line=1 )
plotter.process_data( rolling_window=100 )
plotter.render_std( labels=["PPO (8x2)"], colors=["c"], styles=['-', '-', '-'] )
