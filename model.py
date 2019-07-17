import torch


def get_model(input_dim, hidden, output_dim, lr):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden, output_dim),
        torch.nn.Softmax(dim=0)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer

def loss_fn(preds, r): 
    # pred is output from neural network
    # r is return (sum of rewards to end of episode)
    return -torch.sum(r * torch.log(preds)) # element-wise multipliy, then sum