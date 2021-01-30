from types import SimpleNamespace
import warnings

from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10
from _01_environment.roboadvenv_v10 import InvestUniverse
from _02_agent.roboadvagent_v10 import RoboAdvisorAgentV10
from _03_bridge.simple_bridge_v10 import SimpleBridgeV10
from _04_loopcontrol.loop_control_v10 import LoopControlV10

HYPERPARAMS = {
    'base_setup': SimpleNamespace(**{

        # env
        'env_reward_average_count'    : 1,        # steps over which the reward is averaged
        'env_start_cash'              : 100_000.0,# initial cash position
        'env_trading_cost'            : 40.0,     # costs per sell and buy trade
        'env_buy_volumne'             : 5_000.0,  # amount for which stocks are bought in buy trade

        # agent
        'agent_device'                    : "cuda",   # cpu or cuda
        'agent_gamma_exp'                 : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'               : 50_000,   # size of replay buffer
        'agent_target_net_sync'           : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'          : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'          : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'         : 10**5,    # simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'               : 2000,     # how many nodes are in the hidden layer
        'agent_hidden_layers'             : 2,        # how many layers shall the hidden layer have
        'agent_dueling_network'           : False,    # shall a dueling  net be used
        'agent_steps_count'               : 1,        # how many steps shall be used between training
        'agent_use_combined_replay_buffer': True,     # shall a combined_replay_buffer be used

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 200,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 200_000.0,# target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    })
}

def create_control(params: SimpleNamespace, config_name) -> LoopControlV10:
    universe = InvestUniverse()
    env = RoboAdvisorEnvV10(universe,
                            reward_average_count   = params.env_reward_average_count,
                            start_cash             = params.env_start_cash,
                            trading_cost           = params.env_trading_cost,
                            buy_volume             = params.env_buy_volumne)

    agent = RoboAdvisorAgentV10(env,
                           devicestr                  = params.agent_device,
                           gamma                      = params.agent_gamma_exp,
                           buffer_size                = params.agent_buffer_size,
                           target_net_sync            = params.agent_target_net_sync,
                           eps_start                  = params.agent_simple_eps_start,
                           eps_final                  = params.agent_simple_eps_final,
                           eps_frames                 = params.agent_simple_eps_frames,
                           hidden_size                = params.agent_hidden_size               ,
                           hidden_layers              = params.agent_hidden_layers             ,
                           dueling_network            = params.agent_dueling_network           ,
                           steps_count                = params.agent_steps_count               ,
                           use_combined_replay_buffer = params.agent_use_combined_replay_buffer,
                           )

    bridge = SimpleBridgeV10(agent=agent,
                             output_actions       = len(universe.get_companies()),
                             output_action_states = 3,
                             optimizer            = params.bridge_optimizer,
                             learning_rate        = params.bridge_learning_rate,
                             gamma                = params.bridge_gamma,
                             initial_population   = params.bridge_initial_population,
                             batch_size           = params.bridge_batch_size,
                             )


    control = LoopControlV10(
        bridge              = bridge,
        run_name            = config_name,
        bound_avg_reward    = params.loop_bound_avg_reward,
        logtb               = params.loop_logtb,
        logfolder           = "./../runs/runv00")

    return control

def run_example(config_name: str):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    control = create_control(HYPERPARAMS[config_name], config_name)
    control.run()

if __name__ == '__main__':
    run_example('base_setup')