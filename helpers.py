import torch
from torch.nn import functional as F
import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def train_model(hyperparams, actor_env, training, metrics, early_stop_target=200., early_stop_threshold=5):

    (epochs, gamma) = hyperparams
    (model, env) = actor_env
    (loss_fn, optimizer) = training
    (losses, durations, average_durations) = metrics

    early_stop_captures = []

    for episode in range(epochs):
        if len(early_stop_captures) >= early_stop_threshold:
            print("stopped early because net has reached target score")
            print(early_stop_captures)
            return episode

        state = env.reset()
        done = False
        t = 0
        obs = []
        actions = []

        while not done:
            pred = model(torch.from_numpy(state).float())
            action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
            next_state, _reward, done, _info = env.step(action)
            obs.append(state)
            actions.append(action)
            state = next_state
            t += 1

        ep_len = len(obs)  # M
        rewards = torch.arange(ep_len, 0, -1).float()
        d_rewards = discount_rewards(rewards, gamma)
        preds = torch.zeros(ep_len)

        for time_step in range(ep_len):
            state = obs[time_step]
            action = int(actions[time_step])
            pred = model(torch.from_numpy(state).float())
            preds[time_step] = pred[action]

        loss = loss_fn(preds, d_rewards)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        durations.append(ep_len)
        average_duration = 0. if len(
            durations) < 100 else np.average(durations[-50:])
        average_durations.append(average_duration)
        print("episode {}, loss: {:.5f}, avg: {:.2f} :: {}".format(
            episode, loss, average_duration, ep_len))

        if average_duration >= early_stop_target:
            early_stop_captures.append(average_duration)


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    gamma_tensor = torch.ones((1, lenr)).new_full((1, lenr), gamma).float()
    powers = torch.arange(lenr).float()
    decay = torch.pow(gamma_tensor, powers).float()
    d_rewards = decay * rewards
    d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-07)
    return d_rewards


def plot_losses(losses, filename='', plotName='Loss', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel(plotName)
    plt.xlabel("Training Steps")
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)


def plot_durations(durations, filename='', plotName='Duration', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(durations)), durations)
    plt.ylabel(plotName)
    plt.xlabel('Episode #')
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)


def save_model(model, filename):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, filename)


def load_model(model, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    if evalMode:
        model.eval()
    else:
        model.train()

    return model


def worker(model, params, counter, worker_index, render=False, train=True, max_eps=300):
    losses = []
    durations = []
    env = gym.make("CartPole-v1")
    env._max_episode_steps = max_eps
    if train:
        optimizer = torch.optim.Adam(
            lr=params['lr'], params=model.parameters())

    for epoch in range(params['epochs']):
        values, logprobs, rewards = run_episode(env, model, render)

        if train:
            loss, actor_loss, critic_loss, eplen = update_params(
                optimizer, values, logprobs, rewards, gamma=params['gamma'])

            losses.append(loss.item())
            durations.append(eplen)
            counter.value += 1

            if epoch % 50 == 0:
                rolling_window = 20
                ave_loss = pd.Series(losses).rolling(rolling_window).mean()[-1:].item()
                ave_duration = pd.Series(durations).rolling(rolling_window).mean()[-1:].item()
                print("Epoch: {}, Loss: {:.2f}, Ep Len: {:.2f}".format(epoch, ave_loss, ave_duration))


def run_episode(env, model, render):
    state = torch.from_numpy(env.reset()).float()
    values, logprobs, rewards = [], [], []
    done = False

    while (done == False):
        policy, value = model(state)

        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob = policy.view(-1)[action]

        values.append(value)
        logprobs.append(logprob)

        state_, reward, done, _ = env.step(action.detach().numpy())

        if render:
            env.render()

        state = torch.from_numpy(state_).float()

        if done:
            reward = -10
            env.reset()

        rewards.append(reward)

    return values, logprobs, rewards


def update_params(optimizer, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    total_return = torch.Tensor([0])

    for reward_index in range(len(rewards)):
        total_return = rewards[reward_index] + gamma * total_return
        Returns.append(total_return)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, actor_loss.sum(), critic_loss.sum(), len(rewards)
