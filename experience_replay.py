'''I / we certify that the code and data in this assignment were generated independently, using only the tools
and resources defined in the course and that I/we did not receive any external help, coaching, or contributions
during the production of this work.'''
from collections import deque, namedtuple
import random
import torch

class Experience:
    
    def _init_(self,batch_size):

        self.memory = deque(maxlen=batch_size)
        self.batch_size = batch_size
        self.exp = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def _len_(self):
        return len(self.memory)

    def log_experience(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).type(torch.float)
        next_states = torch.from_numpy(next_states).type(torch.float)
        actions = torch.tensor(actions).type(torch.float)
        rewards = torch.tensor(rewards).type(torch.float)
        dones=torch.tensor(dones)
        experience = self.exp(states, actions, rewards, next_states, dones)
        self.memory.append(experience)
   
    def sample(self):
        experiences=random.sample(self.memory, self.batch_size)
        states = torch.stack([e.state for e in experiences if e is not None])
        actions = torch.stack([e.action for e in experiences if e is not None])
        rewards = torch.stack([e.reward for e in experiences if e is not None])
        next_states = torch.stack([e.next_state for e in experiences if e is not None])
        print(experiences[0].done)
        dones = torch.stack([e.done for e in experiences if e is not None])
        return states, actions, rewards, next_states, dones