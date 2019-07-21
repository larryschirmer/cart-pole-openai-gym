import torch
import gym
import numpy as np
from time import sleep

from model import get_model
from helpers import load_model

lr = 0.0009
input_dim = 4
hidden = 150
output_dim = 2
max_timesteps = 3000

model, optimizer = get_model(input_dim, hidden, output_dim, lr)

filename = 'checkpoint-863.pt'
model, optimizer = load_model(model, optimizer, filename)

env = gym.make('CartPole-v0')
env._max_episode_steps = max_timesteps

done = False
state = env.reset()
duration = 0

while not done:
    pred = model(torch.from_numpy(state).float())
    action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
    next_state, _reward, done, _info = env.step(action)
    env.render()

    state = next_state
    duration += 1

env.close()
print("success: {} :: {:.2f}".format(duration ==
                                 max_timesteps, duration / max_timesteps))
 