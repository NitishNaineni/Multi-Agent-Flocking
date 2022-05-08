import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from boids import parallel_env as penv
from boids import config
from ddpg import DDPG
from replay import Prioritized_Experience_Replay as PER

#def train(env,modelname,mode,epochs,tsteps,tauupdate,batchsize,gamma,hnodes,policylrate,clrate):
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

def collect_experience(env):
    obs=env.observation_spaces
    print(obs)
    for key in obs:
        print(obs[key])
        print("hi")



if __name__ == "__main__":
    env = penv(config=config)
    collect_experience(env)
    '''for i in range(epochs):
        if(i%20==0):
            train(env,agent='agent')
        elif(i%21==0):
            train(env,agent='adversary')
        else:
            collect_experience(env)'''
