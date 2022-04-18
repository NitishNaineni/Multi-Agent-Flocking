# Importing pytorch libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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





