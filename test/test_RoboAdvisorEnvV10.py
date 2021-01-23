from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10


def test_init():
    env = RoboAdvisorEnvV10()
    result = env.reset()

    assert result[0] == 0.0
