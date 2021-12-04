import numpy as np
import random
from collections import deque


class Memory:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, option, reward, state_next, done):
        experience = (state, action, option, reward, state_next, done)
        self.data.append(experience)
        if self.pointer < len(self.data):
            self.pointer += 1

    def sample(self, batch):
        states = np.array([self.data[i][0] for i in batch])
        actions = np.array([self.data[i][1] for i in batch])
        options = np.array([self.data[i][2] for i in batch])
        rewards = np.array([self.data[i][3] for i in batch])
        states_next = np.array([self.data[i][4] for i in batch])
        dones = np.array([self.data[i][5] for i in batch])

        return states, actions, options, rewards, states_next, dones

    def __str__(self):
        memory_state = ""
        for s, a, r, sn, done in self.data:
            if isinstance(s, list):
                # probably agents 2+
                for i in s:
                    memory_state += "{},".format(i.shape)
                memory_state += ";"
            else:
                memory_state += "{};".format(s.shape)
        return memory_state