from gym.envs.registration import register 

register( id='discrete_navigation-v0', entry_point='light_safety_gym.point_navigation:DiscreteNavigation', ) 
register( id='continuous_navigation-v0', entry_point='light_safety_gym.point_navigation:ContinuousNavigation', ) 