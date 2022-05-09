from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from scenario import Scenario
import numpy as np

class raw_env(SimpleEnv):
    def __init__(self, config, continuous_actions=True):
        scenario = Scenario(config)
        world = scenario.make_world()
        super().__init__(scenario, world, config['max_cycles'], continuous_actions)
        self.metadata['name'] = "boids"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

config = {
    
    "shape":True,
    "max_cycles" : 1000,
    "num_good" : 3,
    "num_advr" : 1,
    "num_obst" : 1,
    
    "good_size" : 0.05,
    "advr_size" : 0.075,
    "obst_size" : 0.2,
    
    "advr_accel" : 3.0,
    "good_accel" : 4.0,
    
    "advr_max_speed" : 1.0,
    "good_max_speed" : 1.0,
    
    "good_color" : np.array([0.35, 0.85, 0.35]),
    "obst_color" : np.array([0.25, 0.25, 0.25]),
    "advr_color" : np.array([0.85, 0.35, 0.35]),
    
    "good_spawn_range" : 1,
    "advr_spawn_range" : 1,
    "obst_spawn_range" : 0.9,
    
    "observation_resolution" : 64,

    "good_col_range" : 2,
    "advr_col_range" : 2

}

