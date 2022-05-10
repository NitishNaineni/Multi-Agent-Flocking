""" 
Reference
Github -  https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb
Wiki - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
 """

import numpy as np
# class ouNoise:

#     def __init__(self, env_actions, mu=0.0, theta=0.0015, max_sigma=0.002, min_sigma=0.001, decay_steps=100000):
#         self.mu = mu
#         self.theta = theta
#         self.sigma = max_sigma
#         self.max_sigma = max_sigma
#         self.min_sigma = min_sigma
#         self.decay_steps = decay_steps
#         self.n_env_actions = env_actions.shape[0]
#         self.action_low = env_actions.low
#         self.action_high = env_actions.high
#         self.gen = np.random.default_rng()

#         self.reset()

#     def reset(self):
#         self.state = np.ones(self.n_env_actions)*self.mu
    
#     def change_state(self):
#         x = self.state
#         d_x = self.theta * (self.mu - x) + self.sigma + self.gen.uniform(size=self.n_env_actions, low=0.0, high=0.25)
#         self.state = x + d_x
#         return self.state

#     def add_noise(self, action, timestep=0):
#         ou_state = self.change_state()
#         self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, timestep/self.decay_steps)
#         return np.clip(action + ou_state, self.action_low, self.action_high)



class ouNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def add_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)