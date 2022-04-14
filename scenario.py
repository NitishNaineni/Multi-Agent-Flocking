import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

class Scenario(BaseScenario):
    
    def __init__(self,num_agents,num_adversaries):
        self.num_agents = num_agents
        self.num_adversaries = num_adversaries
        self.total_agents = num_agents + num_adversaries
        self.agent_size = 0.15
        self.adversary_size = 0.3
        self.agent_color = np.array([0.35, 0.35, 0.85])
        self.adversary_color = np.array([0.85, 0.35, 0.35])
        self.adversary_visibility = 2
        self.agent_visibility = 1
        
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.num_agents = self.total_agents
        
        # add agents
        world.agents = [Agent() for i in range(self.total_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < self.num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < self.num_adversaries else i - self.num_adversaries
            agent.name = f'{base_name}_{base_index}'
            agent.collide = True
            agent.silent = True
            agent.size = self.adversary_size if agent.adversary else self.agent_size
        return world
    
    def reset_world(self, world, np_random):
        for i in range(self.total_agents):
            if world.agents[i].adversary:
                world.agents[i].color = self.adversary_color
            else:
                world.agents[i].color = self.agent_color
                
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
    def benchmark_data(self, agent, world):
        # return data for bechmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]
    
    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
    
    def agent_reward(self, agent, world):
        raise NotImplementedError()
        
    def adversary_reward(self, agent, world):
        raise NotImplementedError()
        
    def observation(self, agent, world):
        agents_pos = []
        visibility_range = self.adversary_visibility if agent.adversary else self.agent_visibility
        
        for other in world.agents:
            if other is agent:
                continue
            relative_pos = other.state.p_pos - agent.state.p_pos
            relative_distance = (relative_pos**2).sum()
            if relative_distance < visibility_range:
                agents_pos.append(relative_pos)
        
        return np.concatenate(agents_pos)