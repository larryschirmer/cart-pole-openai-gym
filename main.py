import numpy as np
import torch
import gym
from torch import optim
import torch.multiprocessing as mp
from time import perf_counter


from model import ActorCritic, loss_fn
from helpers import discount_rewards, train_model, plot_losses, plot_durations, save_model, load_model, worker

lr = 0.0015
gamma = 0.99
gae = 0.9
clc = 0.1
step_update = 100
ppo_epsilon = 0.2

input_dim = 4
shared_hidden0 = 25
shared_hidden1 = 50
critic_hidden = 25
output_dim_actor = 2
output_dim_critic = 1

model = ActorCritic(
    input_dim, shared_hidden0, shared_hidden1, critic_hidden, output_dim_actor, output_dim_critic)

env = gym.make('CartPole-v0')

epochs = 500
losses = []
actor_losses = []
critic_losses = []
durations = []

params = {
    'epochs': epochs,
    'n_workers': mp.cpu_count(),
    'lr': lr,
    'step_update': step_update,
    'gamma': gamma,
    'gae': gae,
    'ppo_epsilon': ppo_epsilon,
    'clc': clc, 
    'losses': losses,
    'durations': durations,
    'actor_losses': actor_losses,
    'critic_losses': critic_losses
}

model.share_memory()
processes = []
counter = mp.Value('i', 0)
for worker_index in range(params['n_workers']):
    p = mp.Process(target=worker, args=(model, params, counter, worker_index), kwargs={'max_eps': 5000})
    p.start()
    processes.append(p)
for p in processes:
    p.join()
for p in processes:
    p.terminate()

print(counter.value, processes[1].exitcode)


save_model(model, 'actor_critic.pt')
