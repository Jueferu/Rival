import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BALL_RADIUS, BLUE_TEAM, BALL_MAX_SPEED

class PossesionReward(RewardFunction):
    def __init__(self, min_dist: float = 200):
        super().__init__()
        self.prevTeamTouch = -1
        self.stacking = 0
        self.min_dist = min_dist
    
    def reset(self, initial_state: GameState):
        self.prevTeamTouch = -1
        self.stacking = 0
    
    def pre_step(self, state: GameState):
        for player in state.players:
            if not player.ball_touched:
                continue

            if self.prevTeamTouch != player.team_num:
                self.prevTeamTouch = player.team_num
                self.stacking = 0
                continue

            self.stacking += 1

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist_to_ball = np.linalg.norm(player.car_data.position - state.ball.position)
        
        if self.prevTeamTouch != player.team_num:
            return 0
        if dist_to_ball > self.min_dist:
            return 0
    
        enemies = [p for p in state.players if p.team_num != player.team_num]
        closest_enemy = min(enemies, key=lambda p: np.linalg.norm(p.car_data.position - state.ball.position))
        closest_enemy_dist = np.linalg.norm(closest_enemy.car_data.position - state.ball.position)
        
        if closest_enemy_dist < dist_to_ball:
            return 0
        
        self.stacking += 1
        reward = self.stacking / 10
        return reward