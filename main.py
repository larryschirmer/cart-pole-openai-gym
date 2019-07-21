import gym
import numpy as np
import torch
from time import perf_counter

from model import get_model, loss_fn
from helpers import discount_rewards, train_model, plot_losses, plot_durations, save_model, load_model

lr = 0.001
gamma = 0.99

input_dim = 4
hidden = 150
output_dim = 2

model, optimizer = get_model(input_dim, hidden, output_dim, lr)

env = gym.make('CartPole-v0')
env._max_episode_steps = 3000
epochs = 10000
losses = []
durations = []
average_durations = []


hyperparams = (epochs, gamma)
actor_env = (model, env)
training = (loss_fn, optimizer)
metrics = (losses, durations, average_durations)

start = perf_counter()
final_episode = train_model(hyperparams, actor_env, training,
                            metrics, early_stop_target=1000., early_stop_threshold=10)
save_model(model, optimizer, 'checkpoint-{}.pt'.format(final_episode))
end = perf_counter()
print((end - start))

plot_losses(losses, 'losses-{}.png'.format(final_episode))
plot_durations(durations, 'durations-{}.png'.format(final_episode))
plot_durations(average_durations,
               'ave-durations-{}.png'.format(final_episode),  plotName='Ave Durations')
