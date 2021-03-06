from _02_agent.base_agent import AgentBase
from _03_bridge.base_bridge import BridgeBase

from typing import Iterable, Tuple, List

from ignite.engine import Engine

from ptan.experience import ExperienceFirstLast

import torch
import torch.nn as nn
from torch.optim import Optimizer


class SimpleBridgeV10(BridgeBase):

    def __init__(self,
                 agent: AgentBase,
                 output_actions: int = 497,
                 output_action_states: int = 3,
                 optimizer: Optimizer = None,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.9,
                 initial_population: int = 1000,
                 batch_size: int = 32
                 ):

        super(SimpleBridgeV10, self).__init__(agent, optimizer, learning_rate, gamma, initial_population, batch_size)
        self.output_actions = output_actions
        self.output_action_states = output_action_states

    def get_sample(self):
        return self.agent.buffer.sample(self.batch_size)

    def process_batch(self, engine:Engine, batchdata):
        self.optimizer.zero_grad()

        loss_v = self._calc_loss(batchdata)

        loss_v.backward()
        self.optimizer.step()

        self.agent.iteration_completed(engine.state.iteration)

        return {
            "loss": loss_v.item(),
            "epsilon": self.agent.selector.epsilon,
        }


    def _calc_loss(self, batch: List[ExperienceFirstLast]):

        states, actions, rewards, dones, next_states = self._unpack_batch(batch)

        states_v      = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v     = torch.tensor(actions).to(self.device)
        rewards_v     = torch.tensor(rewards).to(self.device)
        done_mask     = torch.BoolTensor(dones).to(self.device)

        actions_v         = actions_v.unsqueeze(-1)
        state_action_vals = self.agent.net(states_v).view(-1, self.output_actions, self.output_action_states).gather(1, actions_v)
        state_action_vals = state_action_vals.squeeze(-1)

        with torch.no_grad():
            next_state_vals            = self.agent.tgt_net.target_model(next_states_v)\
                                            .view(-1, self.output_actions, self.output_action_states).max(2)[0]
            next_state_vals[done_mask] = torch.zeros(self.output_actions, device=self.device)

        bellman_vals = next_state_vals.detach().sum(1) * self.gamma + rewards_v
        state_action_vals_sum = state_action_vals.sum(1)
        return nn.MSELoss()(state_action_vals_sum, bellman_vals)