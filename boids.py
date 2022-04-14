from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from scenario import Scenario

class raw_env(SimpleEnv):
    def __init__(self, num_agents,num_adversaries, max_cycles=25, continuous_actions=True):
        scenario = Scenario(num_agents,num_adversaries)
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "boids"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)