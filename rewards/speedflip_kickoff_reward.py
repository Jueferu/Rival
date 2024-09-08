import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

#https://github.com/redd-rl/apollo-cpp/blob/main/CustomRewards.h#L18
class SpeedflipKickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_vel = state.ball.linear_velocity
        ball_vel_mag = np.linalg.norm(ball_vel)
        boost_amount = player.boost_amount
        
        if (ball_vel_mag == 0 and boost_amount < 0.02):
            player_vel = player.car_data.linear_velocity
            player_to_ball = state.ball.position - player.car_data.position
            player_to_ball_unit = player_to_ball / np.linalg.norm(player_to_ball)
            player_vel_dot = np.dot(player_vel, player_to_ball_unit)
            
            return max(0, player_vel_dot / CAR_MAX_SPEED)
    
        return 0