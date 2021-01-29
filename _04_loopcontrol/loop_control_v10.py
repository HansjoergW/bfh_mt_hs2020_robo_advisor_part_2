from _03_bridge.base_bridge import BridgeBase
from _04_loopcontrol.base_loop_control import LoopControlBase, TimeHandler, update_smoothed_metrics

from datetime import timedelta, datetime

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler

from ptan.ignite import EndOfEpisodeHandler, EpisodeEvents, PeriodicEvents, PeriodEvents


class LoopControlV10(LoopControlBase):

    # in our default setup, bound_avg_reward of 300_000 would mean me gained about 200%
    def __init__(self, bridge: BridgeBase, run_name: str, bound_avg_reward: float = 300_000.0,
                 logtb: bool = False,
                 logfolder: str = "runs"):
        super(LoopControlV10, self).__init__(bridge, run_name, bound_avg_reward, logtb, logfolder)

        # this handler has several problems. it does more than one thing and it also
        # has to have direct access to the experienceSource of the agent.
        # that could be refactored
        EndOfEpisodeHandler(self.bridge.agent.exp_source, bound_avg_reward=bound_avg_reward).attach(self.engine)
        TimeHandler().attach(self.engine)

        RunningAverage(output_transform=lambda v: v['loss']).attach(self.engine, "avg_loss")
        PeriodicEvents().attach(self.engine)  # creates periodic events

        @self.engine.on(EpisodeEvents.EPISODE_COMPLETED)
        def episode_completed(trainer: Engine):
            self.episode_completed_basic(trainer)
            reward = trainer.state.metrics['reward']
            steps = trainer.state.metrics['steps']

            value = self.bridge.agent.env.get_current_value()

            trainer.state.metrics['value'] = value

            update_smoothed_metrics(0.98, trainer,
                                    ('avg_reward', 'avg_steps', 'avg_value'),
                                    (reward, steps, value))

        @self.engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
        def game_solved(trainer: Engine):
            self.game_solved_basic(trainer)

        if self.logtb:
            handler = OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
            self.tblogger.attach(self.engine, log_handler=handler, event_name=EpisodeEvents.EPISODE_COMPLETED)

            handler = OutputHandler(tag="episodes2", metric_names=['value'])
            self.tblogger.attach(self.engine, log_handler=handler, event_name=EpisodeEvents.EPISODE_COMPLETED)

            handler = OutputHandler(tag="episodes3", metric_names=['avg_value'])
            self.tblogger.attach(self.engine, log_handler=handler, event_name=EpisodeEvents.EPISODE_COMPLETED)

            handler = OutputHandler(tag="train", metric_names=['avg_loss'], output_transform=lambda a: a)
            self.tblogger.attach(self.engine, log_handler=handler, event_name=PeriodEvents.ITERS_100_COMPLETED)
