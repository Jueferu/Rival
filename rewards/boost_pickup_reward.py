import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

#https://github.com/redd-rl/apollo-cpp/blob/main/CustomRewards.h#L204
class BoostPickupReward(RewardFunction):
    def __init__(self, small=3, big=10):
        super().__init__()
        self.last_state = None
        self.small = small
        self.big = big

    def reset(self, initial_state: GameState):
        self.last_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        
        last_state = self.last_state
        for _player in last_state.players:
            if player.car_id == _player.car_id:
                boostDiff = player.boost_amount - _player.boost_amount
                if boostDiff > 0:
                    reward += boostDiff * self.big
                    if (player.boost_amount < 0.98 and _player.boost_amount < 0.88):
                        reward += boostDiff * self.small
                    
        
        self.last_state = state
        return reward