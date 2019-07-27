import numpy as np
import torch
import gym
from torch import optim
import torch.multiprocessing as mp
import pandas as pd
from time import perf_counter


from model import ActorCritic, loss_fn
from helpers import discount_rewards, train_model, plot_losses, plot_durations, save_model, load_model, worker

lr = 0.001
gamma = 0.99

input_dim = 4
shared_hidden = 150
critic_hidden = 25
output_dim_actor = 2
output_dim_critic = 1

model = ActorCritic(
    input_dim, shared_hidden, critic_hidden, output_dim_actor, output_dim_critic)

env = gym.make('CartPole-v0')

epochs = 1000
losses = []
actor_losses = []
critic_losses = []
durations = []

params = {
    'epochs': epochs,
    'n_workers': mp.cpu_count(),
    'lr': lr,
    'gamma': gamma,
    'losses': losses,
    'durations': durations,
    'actor_losses': actor_losses,
    'critic_losses': critic_losses
}


worker(model, params)
save_model(model, 'actor_critic.pt')

rolling_window = 100
ave_loss = pd.Series(losses).rolling(rolling_window).mean()
plot_losses(ave_loss, filename='losses.png', plotName='Losses')

ave_actor_loss = pd.Series(actor_losses).rolling(rolling_window).mean()
plot_losses(actor_losses, filename='actor_losses.png', plotName='Actor Losses')

ave_critic_loss = pd.Series(critic_losses).rolling(rolling_window).mean()
plot_losses(ave_critic_loss, filename='critic_losses.png',
            plotName='Critic Losses')

ave_duration = pd.Series(durations).rolling(50).mean()
plot_durations(ave_duration, filename='durations.png', plotName='Durations')
