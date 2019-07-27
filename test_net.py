import torch
import gym
import numpy as np
from time import sleep

from model import ActorCritic
from helpers import load_model, worker

lr = 0.001
gamma = 0.99

input_dim = 4
shared_hidden = 150
critic_hidden = 25
output_dim_actor = 2
output_dim_critic = 1
max_timesteps = 3000

model = ActorCritic(
    input_dim, shared_hidden, critic_hidden, output_dim_actor, output_dim_critic)

filename = 'actor_critic.pt'
model = load_model(model, filename)

params = {
    'epochs': 1,
    'n_workers': 0
}

worker(model, params, None, 0, render=True, train=False)