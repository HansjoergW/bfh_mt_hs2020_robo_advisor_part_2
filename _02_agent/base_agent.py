from gym import Env

from abc import ABC, abstractmethod

from gym import Env

import torch
import torch.nn as nn
import ptan

class AgentBase(ABC):

    def __init__(self, env: Env, devicestr:str):
        self.env = env
        self.device = torch.device(devicestr)

    @abstractmethod
    def get_net(self):
        pass

    @abstractmethod
    def get_tgtnet(self):
        pass

    @abstractmethod
    def get_buffer(self):
        pass

    @abstractmethod
    def iteration_completed(self, iteration: int):
        pass


def create_net(input_size:int, output_size:int, hidden_layers:int, hidden_size: int) -> nn.Sequential:
    modules = []
    modules.append(nn.Linear(input_size, hidden_size))
    modules.append(nn.ReLU())

    for i in range(0, hidden_layers-1):
        modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*modules)


class SimpleNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_layers, hidden_size):
        super(SimpleNet, self).__init__()
        self.net = create_net(obs_size, n_actions, hidden_layers, hidden_size)

    def forward(self, x):
        return self.net(x.float())


class DuelingNet(nn.Module):
    # gemäss https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751

    def __init__(self, obs_size, n_actions, hidden_layers, hidden_size):
        super(DuelingNet, self).__init__()

        self.feauture_layer = create_net(obs_size, hidden_size, hidden_layers, hidden_size)

        self.value_stream = create_net(hidden_size, 1, hidden_layers, hidden_size)

        self.advantage_stream = create_net(hidden_size, n_actions, hidden_layers, hidden_size)


    def forward(self, x):
        features = self.feauture_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class CombinedReplayBuffer(ptan.experience.ExperienceReplayBuffer):
    # https://arxiv.org/pdf/1712.01275.pdf

    def __init__(self, experience_source, buffer_size):
        super(CombinedReplayBuffer, self).__init__(experience_source, buffer_size)
        self.last_added = None

    def _add(self, sample):
        self.last_added = sample
        super()._add(sample)

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        if self.last_added is not None:
            batch[0] = self.last_added

        return batch
