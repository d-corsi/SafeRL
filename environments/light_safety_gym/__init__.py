from gym.envs.registration import register 

register( id='point_discrete-v0', entry_point='light_safety_gym.base_navigation:PointNavigationDiscrete', ) 
register( id='point_continuous-v0', entry_point='light_safety_gym.base_navigation:PointNavigationContinuous', ) 