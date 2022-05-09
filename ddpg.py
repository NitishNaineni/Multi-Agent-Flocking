from models import Actor, Critic
import torch.nn as nn
import torch.optim as optim
import copy
import torch
from replay import Experience
import numpy as np
class DDPG:
    def __init__(self, n_obs, n_actions, args):

        self.n_actions = n_actions
        self.n_obs = n_obs
        
        self.actor = Actor(self.n_obs, self.n_actions)
        self.target_actor = Actor(self.n_obs, self.n_actions)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr= args.lr_actor)
        
        self.critic = Critic(self.n_obs, self.n_actions)
        self.target_critic = Critic(self.n_obs, self.n_actions)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr= args.lr_critic)

        # Doing a hard update to make sure the parameters are same

        self.hardUpdate()

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.dist_factor = args.disc_factor
        self.epsilon = 1.0
        self.criterion = nn.MSELoss()
        self.args = args

        # if args.CUDA: self.cuda_port()

    # Function for updating policy
    def policyUpdate(self, exp,buffer_size):

        # Sampling the batch from experience replay (Need to change this later on as per the Nitish's reply class)
        instances,avg_prob=exp.sample()
        batch=Experience(*zip(*instances))
        states_batch=torch.tensor(np.array(batch.state, dtype=np.float32))
        actions_batch=torch.tensor(np.array(batch.action, dtype=np.float32))
        rewards_batch=torch.tensor(np.array(batch.reward, dtype=np.float32)).unsqueeze(1)
        nstates_batch=torch.tensor(np.array(batch.next_state, dtype=np.float32))
        done_batch=torch.tensor(np.array(batch.done), dtype=int).unsqueeze(1)
        # print("Type ",batch.state)
        # Calculcating Target Q
        target_actions = self.target_actor(nstates_batch)
        
        
        target_critic_q_values =  self.target_critic(nstates_batch, target_actions) 
        
        target_q_values = rewards_batch + self.dist_factor*(1- done_batch)*target_critic_q_values
        # print("Target Q Values ", target_q_values)
        # Current Q Value
        q_values = self.critic(states_batch, actions_batch)
        # print(" Q Values ", q_values)

        # Calculating the critic loss
        # print(nn.MSELoss(q_values, target_q_values.detach()))
        critic_loss_mse =  nn.MSELoss()
        critic_loss = critic_loss_mse(q_values, target_q_values.detach())/(avg_prob*buffer_size)

        # Critic network Update
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Calculating the policy network loss
        policy_loss = -(self.critic(states_batch, self.actor(states_batch)))
        # policy_loss = self.critic(states_batch, self.actor(states_batch))

        policy_loss = policy_loss.mean()/(avg_prob*buffer_size)

        # Actor Network Update
        self.actor.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


        # Updating the target network parameters softly
        self.softUpdate()
        


    def eval_network(self):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()

    # Function for hard copying parameters of the main network to target network
    def hardUpdate(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        

    # Function for Soft copying parameters of the main network to target network
    def softUpdate(self):
        
        for target_params, params in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.args.tau) + params.data * self.args.tau)

        for target_params, params in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.args.tau) + params.data * self.args.tau)
        


    # This can be implemeneted in a better way later on
    def saveModel(self,name):
        if name=='agent':
            torch.save(self.actor.state_dict(), "agent_actor_params.pt")
            torch.save(self.critic.state_dict(), "agent_critic_params.pt")
        else:
            torch.save(self.actor.state_dict(), "adversary_actor_params.pt")
            torch.save(self.critic.state_dict(), "dversary_critic_params.pt")
        return 

    def LoadModel():
        
        return
    

    def get_actions(self, obs):
        actions = self.actor(torch.from_numpy(obs))
        return actions

    def get_q_value(self, obs, actions):
        q_val = self.critic(obs,actions)
        return q_val

    def get_loss(self,obs,action,next_obs):
        with torch.no_grad():
            self.critic.eval()
            self.actor.eval()
            q_val=self.critic(torch.from_numpy(obs),torch.from_numpy(action))
            next_action=self.actor(torch.from_numpy(next_obs))
            self.target_critic.eval()
            target_q=self.target_critic(torch.from_numpy(next_obs),next_action)
            loss=(target_q-q_val)**2
        return loss