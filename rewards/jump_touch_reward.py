#https://github.com/redd-rl/apollo-cpp/blob/main/CustomRewards.h#L35

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BALL_RADIUS, CEILING_Z, CAR_MAX_SPEED

class JumpTouchReward(RewardFunction):
    def __init__(self, minHeight: int = 200, exp: float = 1) -> None:
        super().__init__()
        
        self.minHeight = minHeight
        self.exp = exp
        self.div = (pow(CEILING_Z / 2 - BALL_RADIUS, exp))
        self.ticksUntilNextReward = 0
        
    def reset(self, game_state: GameState) -> None:
        self.ticksUntilNextReward = 0
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.ball_touched:
            self.ticksUntilNextReward -= 1
            return 0
        
        if not player.on_ground:
            self.ticksUntilNextReward -= 1
            return 0
        
        ball_pos = state.ball.position
        ball_y = ball_pos[2]
        
        if ball_y < self.minHeight:
            self.ticksUntilNextReward -= 1
            return 0

        if self.ticksUntilNextReward > 0:
            self.ticksUntilNextReward -= 1
            return 0

        self.ticksUntilNextReward = 149
        reward = pow(min(ball_y, CEILING_Z / 2) - BALL_RADIUS, self.exp) / self.div
        if reward <= 0.05:
            # why?
            return 0
        
        return reward * 100