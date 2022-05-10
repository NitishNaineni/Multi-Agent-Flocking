'''I / we certify that the code and data in this assignment were generated independently, using only the tools
and resources defined in the course and that I/we did not receive any external help, coaching, or contributions
during the production of this work.'''
import numpy as np
from collections import deque,namedtuple
import random
from torch import float32

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class Prioritized_Experience_Replay:
    def __init__(self, size, alpha, sample_size):
        self.size = size
        self.alpha = alpha
        self.sample_size = sample_size
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.offset = 1
        self.p_total = 0
        self.timestep = 0

    def push(self,loss,*args):
        p = (abs(loss) + self.offset)**self.alpha
        
        if self.timestep >= self.size:
            last = self.priorities[0]
            self.p_total -= last

        self.priorities.append(float(p))
        self.buffer.append(Experience(*args))

        self.p_total += p
        self.timestep += 1

    def sample(self):
        seed = np.random.randint(0,2**31)
        # print("Priorities Type ",type(self.priorities))
        # print("p_total Type ",type(self.p_total))
        sample_prob = list(np.array(self.priorities)/float(self.p_total))
        # print("Buffer ",len(self.buffer))
        # print("Sample prob ",len(sample_prob))
        np.random.seed(seed)
        sampled = random.choices(population=self.buffer,k=self.sample_size,weights=sample_prob)
        np.random.seed(seed)
        priorities = random.choices(population=self.priorities,k=self.sample_size,weights=sample_prob)
        # print("Prior ", priorities)
        # print("p_total Type ",self.p_total)
        avg_prob=np.mean(priorities)/float(self.p_total)
        # print("Priorities Type ",type(priorities))
        # print("p_total Type ",type(self.p_total))
        return sampled,avg_prob

    def is_full(self):
        return self.timestep >= self.size
