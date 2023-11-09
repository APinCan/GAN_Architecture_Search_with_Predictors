import random
import numpy as np

import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.buffer_arch = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity


    def push_predictor(self, arch, n_vertics, adjacent_matrix, operation_matrix, layer, is_score):
        for i in range(self.position):
            if self.buffer[i] is not None:
                b_arch, _, _, _, _, b_is_score = self.buffer[i]
                if b_arch==arch and b_is_score==is_score:
                    return 0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (arch, n_vertics, adjacent_matrix, operation_matrix, layer, is_score)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_predictor(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        arch, n_vertics, adjacent_matrix, operation_matrix, layer, is_score = map(np.stack, zip(*batch))

        return arch, n_vertics, adjacent_matrix, operation_matrix, layer, is_score
    

    def __len__(self):
        return len(self.buffer)
