from _02_agent.base_agent import AgentBase

from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List
import numpy as np

from ignite.engine import Engine
from ptan.experience import ExperienceFirstLast
from torch.optim import Optimizer, Adam

class BridgeBase(ABC):

    def __init__(self, agent: AgentBase, optimizer: Optimizer = None,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.9,
                 initial_population: int = 1000,
                 batch_size: int = 32):
        self.agent = agent
        self.device = agent.device

        self.gamma = gamma
        self.initial_population = initial_population
        self.batch_size = batch_size

        if optimizer is not None:
            self.optimzer = optimizer
        else:
            self.optimizer = Adam(self.agent.net.parameters(), lr=learning_rate)


    def batch_generator(self):
        self.agent.buffer.populate(self.initial_population)
        while True:
            self.agent.buffer.populate(1)
            yield self.get_sample()


    def _unpack_batch(self, batch: List[ExperienceFirstLast]):
        states, actions, rewards, dones, last_states = [],[],[],[],[]

        for exp in batch:
            state = np.array(exp.state)
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)

            if exp.last_state is None:
                lstate = state  # the result will be masked anyway
            else:
                lstate = np.array(exp.last_state)
            last_states.append(lstate)

        return np.array(states, copy=False), \
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones,   dtype=np.uint8), \
               np.array(last_states, copy=False)


    @abstractmethod
    def get_sample(self):
        pass

    @abstractmethod
    def process_batch(self, engine: Engine, batchdata):
        pass
