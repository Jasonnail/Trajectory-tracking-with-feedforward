import torch
import torch.nn as nn


class Controller:
    def __init__(self, path, prefix, s_dim=2, a_dim=1, hidden=32):
        self.actor = DeterministicActor(s_dim, a_dim, hidden) if prefix == 'XY' \
            else DeterministicActor(s_dim, a_dim, hidden)
        self.actor.load_state_dict(torch.load(path + '/'+prefix+'Actor.pth'))

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s)
        return a.item()


class DeterministicActor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, a_dim, bias=False),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)


# class AxiDeterministicActor(nn.Module):
#     def __init__(self, s_dim,  a_dim, hidden):
#         super(AxiDeterministicActor, self).__init__()
#         self.feature_h = nn.Sequential(nn.Linear(2, int(hidden/2)),
#                                        nn.Tanh())
#         self.feature_ev = nn.Sequential(nn.Linear(4, int(hidden/2), bias=False),
#                                         nn.Tanh())
#         self.actor = nn.Sequential(nn.Linear(hidden, hidden, bias=False),
#                                    nn.Tanh(),
#                                    nn.Linear(hidden, hidden, bias=False),
#                                    nn.Tanh(),
#                                    nn.Linear(hidden, 1, bias=False),
#                                    nn.Tanh())
#
#     def forward(self, s):
#         s_h = s[0: 2]
#         s_ev = s[2:]
#         feature_h = self.feature_h(s_h)
#         feature_ev = self.feature_ev(s_ev)
#         a = self.actor(torch.cat([feature_ev, torch.multiply(feature_h, feature_ev)], dim=-1))
#         return a