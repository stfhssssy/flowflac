import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Independent
from typing import Callable, Optional, Union, Tuple, Sequence
from torch.autograd import grad  # Import the gradient function
import torch.distributions as dist
import time

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# define a policy flow v(s, t; \theta)
class Policy_flow(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, steps,action_space=None):
        super(Policy_flow, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs + num_actions + 1, hidden_dim)  # add time embedding, now, time_embedding = time
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        self.steps = steps  # num of steps
        self.device = torch.device(f"cuda:0")
        self.apply(weights_init_)
         
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).cuda()
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).cuda()
            
    def forward(self, state, action_0, time):
        x = torch.cat([state, action_0, time], 1)
        x = self.linear1(x)
        x = self.LayerNorm(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = self.LayerNorm2(x)
        x = F.elu(x)
        x = self.linear3(x)
        return x
    

    def step(self, state, action,  time_start, time_end):
        """
        Integrate the velocity field from time_start to time_end using midpoint method
        Calculate kinetic energy for LAC algorithm
        """
        velocity_start = self.forward(state, action, time_start)
        intermediate_state = action + velocity_start * (time_end - time_start)/2

        velocity_mid = self.forward(state, intermediate_state, time_start + (time_end - time_start)/2)
        action_t = action + velocity_mid * (time_end - time_start)

        # Calculate kinetic energy: 0.5 * ||v||^2 * dt
        step_energy = 0.5 * torch.sum(velocity_mid**2, dim=-1, keepdim=True) * (time_end - time_start)

        return action_t, step_energy
    

    @torch.compile
    def sample(self, state):
        # sampel an action from the nomarl, mean = 0, std = 1
        time_start = torch.zeros(state.shape[0], 1, device = self.device)
        time_step = 1.0 / self.steps  # Assuming we go from t=0 to t=1 in `steps` steps
        action = torch.normal(0, 1, size=(state.shape[0], self.num_actions),device = self.device)
        action = torch.clamp(action,-1.0,1.0)

        # Accumulate total kinetic energy
        total_kinetic = torch.zeros(state.shape[0], 1, device=self.device)

        for i in range(self.steps):
            time_end = time_start + time_step
            action, step_energy = self.step(state, action, time_start, time_end)
            total_kinetic = total_kinetic + step_energy
            time_start = time_end

        # Store raw action before tanh for potential use
        raw_action = action.clone()
        # action = torch.clamp(action,-1.0,1.0)
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        return action, total_kinetic, raw_action

    @torch.compile
    def sample_env(self, state):
        # sampel an action from the nomarl, mean = 0, std = 1
        time_start = torch.zeros(state.shape[0], 1, device=self.device)
        time_step = 1.0 / self.steps  # Assuming we go from t=0 to t=1 in `steps` steps
        action = torch.normal(0, 1, size=(state.shape[0], self.num_actions),device=self.device)
        action = torch.clamp(action,-1.0,1.0)

        # Accumulate total kinetic energy (for consistency, though not used in env sampling)
        total_kinetic = torch.zeros(state.shape[0], 1, device=self.device)

        for i in range(self.steps):
            time_end = time_start + time_step
            action, step_energy = self.step(state, action, time_start, time_end)
            total_kinetic = total_kinetic + step_energy
            time_start = time_end

        raw_action = action.clone()
        #action = torch.clamp(action,-1.0,1.0)
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        return action, total_kinetic, raw_action
