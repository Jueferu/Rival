import numpy as np
from typing import Optional
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

MAX_REWARD = 50

class HitBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        car_velocity = player.car_data.linear_velocity
        
        # car face ball
        car_to_ball = ball_pos - car_pos
        car_forward = player.car_data.forward()
        car_face_ball = np.dot(car_to_ball, car_forward)
        car_face_ball_reward = np.max([0, car_face_ball])
        
        # car speed towards ball
        car_speed = np.linalg.norm(car_velocity)
        dist_to_ball = np.linalg.norm(car_to_ball)
        dir_to_ball = car_to_ball / dist_to_ball
        speed_toward_ball = np.dot(car_to_ball, dir_to_ball)
        speed_toward_ball_reward = np.max([0, speed_toward_ball / CAR_MAX_SPEED])
        
        # hit ball reward
        hit_ball_reward = 1 if player.ball_touched else 0
        
        # air reward
        air_reward = 1 if not player.on_ground else 0
        
        # weigh rewards
        hit_ball_reward *= 50
        car_face_ball_reward *= 1
        speed_toward_ball_reward *= 5
        air_reward *= 0.15
        
        combined_reward = hit_ball_reward + car_face_ball_reward + speed_toward_ball_reward + air_reward
        normalized_reward = combined_reward / MAX_REWARD
        return normalized_reward