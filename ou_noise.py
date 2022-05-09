""" 
Reference
Github -  https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb
Wiki - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
 """

import numpy as np
class ouNoise:

    def __init__(self, env_actions, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_steps=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_steps = decay_steps
        self.n_env_actions = env_actions.shape[0]
        self.action_low = env_actions.low
        self.action_high = env_actions.high-0.05
        self.reset()

    def reset(self):
        self.state = np.ones(self.n_env_actions)*self.mu
    
    def change_state(self):
        x = self.state
        d_x = self.theta * (self.mu - x) + self.sigma + np.random.rand(self.n_env_actions)
        self.state = x + d_x
        return self.state

    def add_noise(self, action, timestep=0):
        ou_state = self.change_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, timestep/self.decay_steps)
        return np.clip(action + ou_state, self.action_low, self.action_high)


