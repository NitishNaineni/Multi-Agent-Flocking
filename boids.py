from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from scenario import Scenario

class raw_env(SimpleEnv):
    def __init__(self, max_cycles=25, continuous_actions=False):
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "boids"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)