import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dim, shared_hidden, critic_hidden, output_dim_actor, output_dim_critic):
        super(ActorCritic, self).__init__()
        self.shared_linear = nn.Linear(input_dim, shared_hidden)

        self.actor_linear = nn.Linear(shared_hidden, output_dim_actor)

        self.critic_linear1 = nn.Linear(shared_hidden, critic_hidden)
        self.critic_linear2 = nn.Linear(critic_hidden, output_dim_critic)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.shared_linear(x))

        actor = F.log_softmax(self.actor_linear(y), dim=0)

        c = F.relu(self.critic_linear1(y.detach()))
        critic = torch.tanh(self.critic_linear2(c))
        return actor, critic


def loss_fn(preds, r):
    # pred is output from neural network
    # r is return (sum of rewards to end of episode)
    return -torch.sum(r * torch.log(preds))  # element-wise multipliy, then sum
