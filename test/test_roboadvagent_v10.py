from _02_agent.roboadvagent_v10 import RoboAdvisorAgentV10
from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10
from _01_environment.universe import InvestUniverse


GAMMA = 0.9
REPLAY_SIZE = 1000
universe = InvestUniverse()

def test_roboagent_cpu():
    print("test cpu")
    env = RoboAdvisorEnvV10(universe)
    agent = RoboAdvisorAgentV10(env, "cpu", gamma=GAMMA, buffer_size=REPLAY_SIZE)
    print(agent.get_net())