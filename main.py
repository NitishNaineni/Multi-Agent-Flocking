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
from gym.spaces import Box
import warnings
warnings.filterwarnings("always")

def collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg,agent_noise,adversary_noise,agent_scores,adversary_scores):
    count=0
    done=False
    score_agent=0
    score_adversary=0
    while (count<=args.timesteps)or (not done):
        actions={}
        loss={}
        for key in obs:
            if(key.find('adversary') != -1):
                temp=adversary_ddpg.get_actions(obs[key])
                temp=adversary_noise.add_noise(temp.detach().numpy(),timestep=count)
                actions[key]=temp.astype(np.float32)

            else:
                temp=agent_ddpg.get_actions(obs[key])
                temp=agent_noise.add_noise(temp.detach().numpy(),timestep=count)
                actions[key]=temp.astype(np.float32)

        nex_obs, reward, done,_= env.step(actions)
        #print("Reward",reward)
        print(count)
        loss=0
        for key in obs:
            if(key.find('adversary') != -1):
                loss=adversary_ddpg.get_loss(obs[key],actions[key],nex_obs[key])
                adversary_per.push(loss,obs[key],actions[key],reward[key],nex_obs[key],done)
                score_agent
            else:
                loss=agent_ddpg.get_loss(obs[key],actions[key],nex_obs[key])
                agent_per.push(loss,obs[key],actions[key],reward[key],nex_obs[key],done)
        obs=nex_obs
        count+=1
    #return score


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
    agent_noise=ouNoise(env.action_spaces['agent_0'],decay_steps=args.timesteps)
    adversary_noise=ouNoise(env.action_spaces['adversary_0'],decay_steps=args.timesteps)
    agent_scores=[]
    adversary_scores=[]
    for i in range(1):
        obs=env.reset()
        if(i%20==0) and (len(agent_per.buffer)>=args.batch_size):
            agent_ddpg.policyUpdate(agent_per)
            #train(env,args,agent_ddpg,agent_per)
            agent_ddpg.saveModel()
        elif(i%21==0) and  (len(adversary_per.buffer)>=args.batch_size):
            adversary_ddpg.policyUpdate(adversary_per)
            #train(env,args,adversary_ddpg,adversary_per)
            adversary_ddpg.saveModel()
        else:
            collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg,agent_noise,adversary_noise,agent_scores,adversary_scores)
        
