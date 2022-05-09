import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent,Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

class Scenario(BaseScenario):
    def __init__(self,config):
        self.num_good = config['num_good']
        self.num_advr = config['num_advr']
        self.num_obst = config['num_obst']
        
        self.good_size = config['good_size']
        self.advr_size = config['advr_size']
        self.obst_size = config['obst_size']
        
        self.advr_accel = config['advr_accel']
        self.good_accel = config['good_accel']
        
        self.advr_max_speed = config['advr_max_speed']
        self.good_max_speed = config['good_max_speed']
        
        self.good_color = config['good_color']
        self.obst_color = config['obst_color']
        self.advr_color = config['advr_color']
        
        self.good_spawn_range = config['good_spawn_range']
        self.advr_spawn_range = config['advr_spawn_range']
        self.obst_spawn_range = config['obst_spawn_range']
        
        self.observation_resolution = config['observation_resolution']
        res = self.observation_resolution
        self.col_lines = np.array([(np.cos(2 * np.pi * i / res),np.sin(2 * np.pi * i / res)) 
                                   for i in range(res)],dtype=np.float16)
        self.good_col_range = config['good_col_range']
        self.advr_col_range = config['advr_col_range']

        self.shape=config['shape']
        
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.num_good
        num_adversaries = self.num_advr
        num_agents = num_adversaries + num_good_agents
        num_landmarks = self.num_obst
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f'{base_name}_{base_index}'
            agent.collide = True
            agent.silent = True
            agent.size = self.advr_size if agent.adversary else self.good_size
            agent.accel = self.advr_accel if agent.adversary else self.good_accel
            agent.max_speed = self.advr_max_speed if agent.adversary else self.good_max_speed  
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = self.obst_size
            landmark.boundary = False
        return world
    
    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = self.good_color if not agent.adversary else  self.advr_color
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = self.obst_color
        # set random initial states
        for agent in world.agents:
            spawn_range = self.advr_spawn_range if agent.adversary else self.good_spawn_range
            agent.state.p_pos = np_random.uniform(-spawn_range, +spawn_range, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                spawn_range = self.obst_spawn_range
                landmark.state.p_pos = np_random.uniform(-spawn_range, +spawn_range, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = self.shape
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = self.shape
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents)
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew
    
    def segment_collisions(self,circle_center,circle_radius):
        c = circle_center
        p1 = - c
        p2 = self.col_lines - c
        d = self.col_lines
        dr = np.expand_dims((d**2).sum(axis=1),1)
        temp = p1 * np.flip(p2,axis=1)
        big_d = np.expand_dims(temp[:,0] - temp[:,1],1)
        det = circle_radius**2 * dr**2 - big_d**2
        sign = -np.sign(d)
        big_d2 = np.hstack([big_d,-big_d])
        pi = c + (big_d2 * np.flip(d,axis=1) - d * np.sign(det) * abs(det)**0.5) / dr**2
        frac = np.choose(abs(d).argmax(axis=1),pi.T)
        selector = (0 <= frac) & (frac <= 1) & (np.squeeze(det) > 0)
        pi[~selector] = self.col_lines[0]
        return np.sqrt((pi**2).sum(axis=1)),selector

    # def observation(self, agent, world):
        
    #     landmark_collisions = np.ones(self.observation_resolution) * (self.advr_col_range if agent.adversary else self.good_col_range) / 2
    #     # get positions of all entities in this agent's reference frame
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             relative_pos = entity.state.p_pos - agent.state.p_pos
    #             collisions,_ = self.segment_collisions(relative_pos,entity.size/2)
    #             landmark_collisions = np.minimum(landmark_collisions,collisions)
                
    #     advr_collisions = np.ones(self.observation_resolution) * (self.advr_col_range if agent.adversary else self.good_col_range) / 2
    #     good_collisions = np.ones(self.observation_resolution) * (self.advr_col_range if agent.adversary else self.good_col_range) / 2
    #     agent_vels = np.zeros((self.observation_resolution,2))
    #     for other in world.agents:
    #         if other is agent:
    #             continue
    #         relative_pos = other.state.p_pos - agent.state.p_pos
    #         collisions,selector = self.segment_collisions(relative_pos,other.size/2)
    #         agent_vels[selector] = other.state.p_vel 
    #         if other.adversary:
    #             advr_collisions = np.minimum(advr_collisions,collisions)
    #         else:
    #             good_collisions = np.minimum(good_collisions,collisions)
    #     return np.hstack([agent.state.p_vel,agent.state.p_pos,good_collisions,advr_collisions,landmark_collisions,agent_vels.flatten()])

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            relative_pos = other.state.p_pos - agent.state.p_pos
            collisions,selector = self.segment_collisions(relative_pos,other.size/2)
            agent_vels[selector] = other.state.p_vel 
            # print(collisions)
            if other.adversary:
                advr_collisions = np.minimum(advr_collisions,collisions)
            else:
                good_collisions = np.minimum(good_collisions,collisions)
        return np.hstack([agent.state.p_vel,agent.state.p_pos,good_collisions,advr_collisions,landmark_collisions,agent_vels.flatten()])
