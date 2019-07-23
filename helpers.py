import torch
from torch.nn import functional as F
import gym
import numpy as np
from matplotlib import pyplot as plt


def train_model(hyperparams, actor_env, training, metrics, early_stop_target=200., early_stop_threshold=5):

    (epochs, gamma) = hyperparams
    (model, env) = actor_env
    (loss_fn, optimizer) = training
    (losses, durations, average_durations) = metrics

    current_episode = 0
    early_stop_captures = []

    for episode in range(epochs):
        if len(early_stop_captures) >= early_stop_threshold:
            print("stopped early because net has reached target score")
            print(early_stop_captures)
            break

        current_episode = episode
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

    return current_episode


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    gamma_tensor = torch.ones((1, lenr)).new_full((1, lenr), gamma).float()
    powers = torch.arange(lenr).float()
    decay = torch.pow(gamma_tensor, powers).float()
    d_rewards = decay * rewards
    d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-07)
    return d_rewards


def plot_losses(losses, filename='', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(np.arange(len(losses)), losses)
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


def save_model(model, optimizer, filename):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)


def load_model(model, optimizer, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if evalMode:
        model.eval()
    else:
        model.train()

    return model, optimizer


def worker(worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env._max_episode_steps = 3000
    worker_env.reset()
    worker_opt = torch.optim.Adam(
        lr=params['lr'], params=worker_model.parameters())

    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(
            worker_opt, values, logprobs, rewards, gamma=params['gamma'])

        counter.value = counter.value + 1


def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0

    while (done == False):
        j += 1
        policy, value = worker_model(state)

        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob = policy.view(-1)[action]

        logprobs.append(logprob)
        values.append(value)
        
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()

        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0

        rewards.append(reward)

    return values, logprobs, rewards


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])

    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    loss.backward()
    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)
