import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from models import ActorCriticPPO

class PPO:
    def __init__(self,num_inputs, num_outputs, hidden_size,lr):
        
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        self.agent_model = ActorCriticPPO(num_inputs, num_outputs, hidden_size).to(self.device)
        self.agent_optimizer = optim.Adam(self.agent_model.parameters(), lr=lr)
        self.adversary_model = ActorCriticPPO(num_inputs, num_outputs, hidden_size).to(self.device)
        self.adversary_optimizer = optim.Adam(self.adversary_model.parameters(), lr=lr)

    def compute_gae(self,next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self,mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self,ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self,envs,max_frames,num_steps,ppo_epochs,mini_batch_size):
        state = envs.reset()
        early_stop = False
        self.test_rewards = []


        while frame_idx < max_frames and not early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1
                
                # if frame_idx % 1000 == 0:
                #     test_reward = np.mean([self.test_env(envs) for _ in range(10)])
                #     self.test_rewards.append(test_reward)
                #     plot(frame_idx, self.test_rewards)
                #     if test_reward > threshold_reward: early_stop = True
                    

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            
            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


