from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])
        return True

    def _on_rollout_end(self) -> None:
        episode_return = sum(self.rewards) / len(self.rewards)
        self.logger.record('episode_return', episode_return)
        
        # FIXXME - just for debug output
        print("episode_return = ", episode_return )
        self.rewards = []
