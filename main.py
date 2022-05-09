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

def collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg,agent_noise,adversary_noise):
    count=0
    DONE=False
    score_agent=0
    score_adversary=0
    while not DONE:
        # print("Iteration Count ",count)
        actions={}
        loss={}
        for key in obs:
            if(key.find('adversary') != -1):
                temp=adversary_ddpg.get_actions(obs[key])
                # print("NN ", temp)
                # temp=adversary_noise.add_noise(temp.detach().numpy(),timestep=count)
                # print("Noise ", temp)
                actions[key]=temp.detach().numpy().astype(np.float32)
                # actions[key] = np.clip(actions[key], env.action_space(key).low, env.action_space(key).high)

            else:
                temp=agent_ddpg.get_actions(obs[key])
                # temp=agent_noise.add_noise(temp.detach().numpy(),timestep=count)
                actions[key]=temp.detach().numpy().astype(np.float32)
                # actions[key] = np.clip(actions[key], env.action_space(key).low, env.action_space(key).high)

        nex_obs, reward, done,_= env.step(actions)
        # print(nex_obs)
        env.render()
        # print(actions)
        loss=0
        for key in obs:
            if(key.find('adversary') != -1):
                loss=adversary_ddpg.get_loss(obs[key],actions[key],nex_obs[key])
                adversary_per.push(loss,obs[key],actions[key],reward[key],nex_obs[key],done[key])
                score_adversary+=reward[key]
            else:
                loss=agent_ddpg.get_loss(obs[key],actions[key],nex_obs[key])
                agent_per.push(loss,obs[key],actions[key],reward[key],nex_obs[key],done[key])
                score_agent+=reward[key]
        obs=nex_obs
        count+=1
        if(count>=args.timesteps or all(x==True for x in done.values())):
            DONE=True
    print("Adversary Reward ", score_adversary)
    print("Agent Reward",score_agent )
    # env.close()
    return score_agent,score_adversary


if __name__ == "__main__":
    env = penv(config=config)
    obs=env.reset()
    num_states=env.observation_space('agent_0').shape[0]
    num_actions=env.action_space('agent_0').shape[0]
    args=parameter_args()
    agent_per=PER(args.buffer_size_agent,args.exp_alpha,args.batch_size)
    adversary_per=PER(args.buffer_size_adversary,args.exp_alpha,args.batch_size)
    agent_ddpg=DDPG(num_states,num_actions,args)
    adversary_ddpg=DDPG(num_states,num_actions,args)
    agent_noise=ouNoise(env.action_space('agent_0'),decay_steps=args.timesteps)
    adversary_noise=ouNoise(env.action_space('adversary_0'),decay_steps=args.timesteps)
    agent_scores=[]
    adversary_scores=[]
    train_agent = True
    train_count = 0
    for i in range(args.epoch):
        print("Current Epoch ",i)
        obs=env.reset()
        train_count += 1
        score_agent,score_adversary= collect_experience(env,obs,args,agent_per,adversary_per,agent_ddpg,adversary_ddpg,agent_noise,adversary_noise)
        agent_scores.append(score_agent)
        adversary_scores.append(score_adversary)
        if train_agent and (len(agent_per.buffer)>=args.batch_size):
            print("Training Agent")
            agent_ddpg.policyUpdate(agent_per,args.buffer_size_agent)
            # if i%5==0
            agent_ddpg.saveModel()
        elif not train_agent and  (len(adversary_per.buffer)>=args.batch_size):
            print("Training Adversary")
            adversary_ddpg.policyUpdate(adversary_per,args.buffer_size_adversary)
            adversary_ddpg.saveModel()
        
           
        if train_count == 10:
            train_agent = not train_agent
            train_count = 0
    env.close()
