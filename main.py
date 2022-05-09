from ast import arg
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from boids import parallel_env as penv
from boids import config
from ddpg import DDPG
from args import parameter_args
from ou_noise import ouNoise
from replay import Prioritized_Experience_Replay as PER

def train(env,agent): 
    #per=PER() #parameters
    if(agent=='agent'):
        num_states=env.observation_spaces['agent_0'].shape[0]
        num_actions=env.action_spaces['agent_0'].shape[0]
        bound_above=env.action_spaces['agent_0'].high
        bound_below=env.action_spaces['agent_0'].low
    elif(agent=="adversary"):
        num_states=env.observation_spaces['adversary_0'].shape[0]
        num_actions=env.action_spaces['adversary_0'].shape[0]
        bound_above=env.action_spaces['adversary_0'].high
        bound_below=env.action_spaces['adversary_0'].low
    ddpg=DDPG(num_states,num_actions,args,bound_above,bound_below)
    for i in range(10):
        score=0
        obs=env.reset()
        done=False
        #while not done:
            #torch.distributions.(probs=ddpg.actor(torch.from_numpy(obs).float()))

def collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg):
    #obs=env.observation_spaces
    count=0
    done=False
    while (count<=args.timesteps)or (not done):
        actions={}
        for key in obs:
            if(key.find('adversary') != -1):
                temp=adversary_ddpg.get_actions(obs[key])
                actions[key]=temp
            else:
                temp=agent_ddpg.get_actions(obs[key])
                actions[key]=temp

        nex_obs, reward, done,_= env.step(actions)

        for key in obs:
            if(key.find('adversary') != -1):
                adversary_per.push(obs[key],actions[key],reward,nex_obs[key],done)
            else:
                agent_per.push(obs[key],actions[key],reward,nex_obs[key],done)
        obs=nex_obs



if __name__ == "__main__":
    env = penv(config=config)
    obs=env.reset()
    num_states=env.observation_spaces['agent_0'].shape[0]
    num_actions=env.action_spaces['agent_0'].shape[0]
    args=parameter_args()
    agent_per=PER(args.buffer_size_agent,args.exp_alpha,args.batch_size)
    adversary_per=PER(args.buffer_size_adversary,args.exp_alpha,args.batch_size)
    agent_ddpg=DDPG(num_states,num_actions,args)
    adversary_ddpg=DDPG(num_states,num_actions,args)
    for i in range(args.epochs):
        obs=env.reset()
        if(i%20==0) and (len(agent_per.buffer)>=args.batch_size):
            train(env,args,agent_ddpg,agent='agent')
        elif(i%21==0) and  (len(adversary_per.buffer)>=args.batch_size):
            train(env,args,adversary_ddpg,agent='adversary')
        else:
            collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg)
