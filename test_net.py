import torch
import gym
import numpy as np
from time import sleep

from model import ActorCritic
from helpers import load_model, worker

lr = 0.001
gamma = 0.99
gae = 0.9
clc = 0.1
step_update = 50
ppo_epsilon = 0.2

input_dim = 4
shared_hidden0 = 25
shared_hidden1 = 50
critic_hidden = 25
output_dim_actor = 2
output_dim_critic = 1

model = ActorCritic(
    input_dim, shared_hidden0, shared_hidden1, critic_hidden, output_dim_actor, output_dim_critic)

filename = 'actor_critic_checkpoint@highest.pt'
# filename = 'actor_critic.pt'
model = load_model(model, filename)

params = {
    'epochs': 1,
    'n_workers': 0,
    'lr': lr
}

worker(model, params, render=True, train=False)