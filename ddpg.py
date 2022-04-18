from models import Actor, Critic
from experience_replay import Experience
import torch.nn as nn
import copy

class DDPG:
    def __init__(self, n_obs, n_actions, args):

        self.n_actions = n_actions
        self.n_obs = n_obs


        self.actor = Actor(self.n_obs, self.n_actions)
        self.target_actor = Actor(self.n_obs, self.n_actions)
        
        self.critic = Critic(self.n_obs, self.n_actions)
        self.target_critic = Critic(self.n_obs, self.n_actions)

        # Doing a hard update to make sure the parameters are same

        self.hardUpdate()
        self.hardUpdate()

        # Random Process noise
        self.random_noise = self.OhNoise()

        # Experience Replay class
        self.exp = Experience(1000)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.dist_factor = args.disc
        self.epsilon = 1.0
        self.criterion = nn.MSELoss()
        

        if args.CUDA: self.cuda_port()

    # Function for updating policy
    def policyUpdate(self):

        # Sampling the batch from experience replay
        states_batch, actions_batch, rewards_batch, nstates_batch, done_batch = self.exp.sample()

        # Calculcating Target Q
        target_actions = self.target_actor(nstates_batch)
        
        target_critic_q_values =  self.target_critic(nstates_batch, target_actions) 
        
        target_q_values = rewards_batch + self.dist_factor*(1- done_batch)*target_critic_q_values

        q_values = self.critic(states_batch, actions_batch)

        critic_loss =  nn.MSELoss(q_values, target_q_values.detach())
        

        # Critic network Update
        self.critic.zero_grad()
        critic_loss.backward()
        
        # Calculating the policy network loss
        policy_loss = -(self.critic(states_batch, self.actor(states_batch)))
        policy_loss = policy_loss.mean()

        # Actor Network Update
        self.actor.zero_grad()
        policy_loss.backward()


        # Updating the target network parameters softly

        self.softUpdate()
        self.softUpdate()

        return 0


    def eval_network(self):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()

    # Function for hard copying parameters of the main network to target network
    def hardUpdate():

        return 0

    # Function for Soft copying parameters of the main network to target network
    def softUpdate():

        return 0


