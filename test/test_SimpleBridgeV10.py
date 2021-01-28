from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10, InvestUniverse
from _02_agent.roboadvagent_v10 import RoboAdvisorAgentV10
from _03_bridge.simple_bridge_v10 import SimpleBridgeV10


from ignite.engine import Engine
from typing import Iterable, Tuple, List
import numpy as np
from ptan.experience import ExperienceFirstLast

universe = InvestUniverse()


def basic_simple_init(devicestr="cpu") -> SimpleBridgeV10:
    env = RoboAdvisorEnvV10(universe)
    agent = RoboAdvisorAgentV10(env, devicestr, gamma=0.9, buffer_size=1000)
    bridge = SimpleBridgeV10(agent, gamma=0.9)

    return bridge


def simple_experiences(agent:RoboAdvisorAgentV10) -> List[ExperienceFirstLast]:
    obs_space_zero = np.zeros(agent.observation_size, dtype=np.float32)
    obs_space_ones = np.ones(agent.observation_size, dtype=np.float32)


    return [
        ExperienceFirstLast(obs_space_zero, np.int64(1), 1.0, obs_space_ones / 2),
        ExperienceFirstLast(obs_space_ones, np.int64(1), 2.0,  None)
    ]


def test_init_cuda():
    assert basic_simple_init("cuda") != None


def test_init_cpu():
    assert basic_simple_init("cpu") != None


def test_unpack():
    bridge = basic_simple_init()
    batch = simple_experiences(bridge.agent)
    unpacked = bridge._unpack_batch(batch)
    # todo -Checks


def test_calc_loss():
    bridge = basic_simple_init()
    batch = simple_experiences(bridge.agent)
    loss = bridge._calc_loss(batch)
    # todo -Checks

def test_process_batch(devicestr="cpu"):
    bridge = basic_simple_init(devicestr)
    batch = simple_experiences(bridge.agent)
    bridge.process_batch(Engine(bridge.process_batch), batch)
    # todo -Checks


def test_batch_generator(devicestr="cuda"):
    # Test Iterator
    bridge = basic_simple_init(devicestr)
    a = bridge.batch_generator()
    nextbatch = next(a)
    assert len(nextbatch) == 32
