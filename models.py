# Importing pytorch libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor():
         
    def __init__(self, n_obs, n_actions, n_hNodes = 64):
        super().__init__()
        
        # Number of hidden nodes
        self.hNodes = n_hNodes

        # Layers
        self.fc1 = nn.Linear(n_obs, self.hNodes)
        self.fc2 = nn.Linear(self.hNodes, self.hNodes)
        self.fc3 = nn.Linear(self.hNodes, self.hNodes)
        self.fc4 = nn.Linear(self.hNodes, n_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))

        return x




class Critic():
     
    def __init__(self, n_obs, n_actions, n_hNodes = 64):
        super().__init__()

        # Layers
        self.fc1 = nn.Linear(n_obs, n_hNodes)
        self.fc2 = nn.Linear(n_hNodes + n_actions, n_hNodes) # Introducing the actions in the hidden layer
        self.fc3 = nn.Linear(n_hNodes , n_hNodes)
        self.fc4 = nn.Linear(n_hNodes, 1)
        
    def forward(self, obs, action):

        x = F.relu(self.fc1(obs))
        x = torch.cat((x, action), 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

class ActorCriticPPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticPPO, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)




