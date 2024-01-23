import torch
import torch.nn as nn


class Controller:
    def __init__(self, path, prefix, s_dim=2, a_dim=1, hidden=32):
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor.load_state_dict(torch.load(path + '/'+prefix+'Actor.pth'))

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s)
        return a.item()


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)

class Actor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(Actor, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                     nn.Tanh(),
                                     nn.Linear(hidden, hidden, bias=False),
                                     nn.Tanh())
        self.mean = nn.Sequential(nn.Linear(hidden, a_dim, bias=False),
                                  nn.Tanh())
        # self.std = nn.Sequential(nn.Linear(hidden, a_dim),
        #                          nn.Softplus())
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)
        init_linear(self)

    def forward(self, s):
        feature = self.feature(s)
        mean = self.mean(feature)
        std = self.log_std.exp()
        return mean