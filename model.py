import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dim, shared_hidden0, shared_hidden1, critic_hidden, output_dim_actor, output_dim_critic):
        super(ActorCritic, self).__init__()
        self.shared_linear0 = nn.Linear(input_dim, shared_hidden0)
        self.shared_linear1 = nn.Linear(shared_hidden0, shared_hidden1)

        self.actor_linear = nn.Linear(shared_hidden1, output_dim_actor)

        self.critic_linear1 = nn.Linear(shared_hidden1, critic_hidden)
        self.critic_linear2 = nn.Linear(critic_hidden, output_dim_critic)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.shared_linear0(x))
        y = F.relu(self.shared_linear1(y))

        actor = F.log_softmax(self.actor_linear(y), dim=0)

        c = F.relu(self.critic_linear1(y.detach()))
        critic = torch.tanh(self.critic_linear2(c))
        return actor, critic


def loss_fn(preds, r):
    # pred is output from neural network
    # r is return (sum of rewards to end of episode)
    return -torch.sum(r * torch.log(preds))  # element-wise multipliy, then sum
