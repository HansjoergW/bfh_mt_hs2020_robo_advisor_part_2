from _02_agent.base_agent import AgentBase, SimpleNet, DuelingNet, CombinedReplayBuffer

import numpy as np
import torch.nn as nn
from gym import Env

import ptan


class RAActionSelector(ptan.actions.ActionSelector):
    """
    Selects actions using argmax
    """

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=2).astype(np.int64)


class RAEpsilonGreedyActionSelector():

    def __init__(self, actions:int, action_states:int, epsilon=0.05):
        """ actions: the number of different traded stocks for which an action has to be picked.pt
            action_states: the number of different possible actions for every stock: buy, do_nothing, sell"""
        self.epsilon = epsilon
        self.selector = RAActionSelector()
        self.actions = actions
        self.action_states = action_states

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, output_nodes = scores.shape
        actions = self.selector(scores.reshape((-1, self.actions, self.action_states)))
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(self.action_states, sum(mask)*self.actions)
        actions[mask] = rand_actions.reshape((-1, self.actions))
        return actions


class RoboAdvisorAgentV10(AgentBase):

    def __init__(self, env: Env,
                 devicestr: str,
                 gamma: float,
                 buffer_size: int,
                 target_net_sync: int = 1000,
                 eps_start: float = 1.0,
                 eps_final: float = 0.02,
                 eps_frames: int = 10 ** 5,
                 hidden_size: int = 128,
                 hidden_layers: int = 1,
                 dueling_network: bool = False,
                 steps_count: int = 1,
                 use_combined_replay_buffer: bool = False):

        super(RoboAdvisorAgentV10, self).__init__(env, devicestr)

        self.action_size = env.action_space.shape[0] * 3
        self.observation_size = env.observation_space.shape[0] * env.observation_space.shape[1]

        self.target_net_sync = target_net_sync

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.net = self._config_dueling_net() if dueling_network else self._config_simple_net()

        print(self.net)

        self.tgt_net = ptan.agent.TargetNet(self.net)

        self.selector = RAEpsilonGreedyActionSelector(
            actions = env.action_space.shape[0],
            action_states = 3,
            epsilon=1,
        )

        self.epsilon_tracker = ptan.actions.EpsilonTracker(selector=self.selector, eps_start=eps_start,
                                                           eps_final=eps_final, eps_frames=eps_frames)

        self.agent = ptan.agent.DQNAgent(self.net, self.selector, device=self.device)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.agent, gamma=gamma,
                                                                    steps_count=steps_count)

        if use_combined_replay_buffer:
            self.buffer = CombinedReplayBuffer(self.exp_source, buffer_size=buffer_size)
        else:
            self.buffer = ptan.experience.ExperienceReplayBuffer(self.exp_source, buffer_size=buffer_size)

    def _config_simple_net(self) -> nn.Module:
        return SimpleNet(self.observation_size, self.action_size,
                         self.hidden_layers, self.hidden_size).to(self.device)

    def _config_dueling_net(self) -> nn.Module:
        return DuelingNet(self.observation_size, self.action_size,
                          self.hidden_layers, self.hidden_size).to(self.device)

    def iteration_completed(self, iteration: int):

        self.epsilon_tracker.frame(iteration)

        if iteration % self.target_net_sync == 0:
            self.tgt_net.sync()

    def get_net(self):
        return self.net

    def get_tgtnet(self):
        return self.tgt_net

    def get_buffer(self):
        return self.buffer
