import numpy as np
import torch
import os
import pickle
import random

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    #non_uniform sample
    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        num_uniform = int(batch_size * 0.8)
        num_recent = batch_size - num_uniform
        uniform_indices = random.sample(range(buffer_len), num_uniform)

        recent_window = min(2048, buffer_len)
        if self.position - recent_window >= 0:
            recent_indices_raw = list(range(self.position - recent_window, self.position))
        else:
            recent_indices_raw = list(range(self.position - recent_window + buffer_len, buffer_len)) + list(range(0, self.position))
        recent_indices = random.sample(recent_indices_raw, min(num_recent, recent_window))

        all_indices = uniform_indices + recent_indices
        batch = [self.buffer[idx] for idx in all_indices]

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


    def sample_uniform(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def sample_all(self):
        state, action,log_prob, reward, next_state, done = map(np.stack, zip(*self.buffer))
        return state, action,log_prob, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path, i_episode):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity