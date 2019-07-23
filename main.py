import numpy as np
import torch
from torch import optim
import torch.multiprocessing as mp
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

model, optimizer = ActorCritic(
    input_dim, shared_hidden, critic_hidden, output_dim_actor, output_dim_critic)


epochs = 10000
losses = []
durations = []
average_durations = []


hyperparams = (epochs, gamma)
actor_env = (model, env)
training = (loss_fn, optimizer)
metrics = (losses, durations, average_durations)

params = {
    'epochs': epochs,
    'n_workers': mp.cpu_count(),
    'lr': lr,
    'gamma': gamma
}


start = perf_counter()

worker(t, worker_model, counter, params)


final_episode = train_model(hyperparams, actor_env, training,
                            metrics, early_stop_target=1000., early_stop_threshold=10)
save_model(model, optimizer, 'checkpoint-{}.pt'.format(final_episode))
end = perf_counter()
print((end - start))

plot_losses(losses, 'losses-{}.png'.format(final_episode))
plot_durations(durations, 'durations-{}.png'.format(final_episode))
plot_durations(average_durations,
               'ave-durations-{}.png'.format(final_episode),  plotName='Ave Durations')
