import numpy as np
import time
import os
import json

def get_checkpoint_value(checkpoint_file: str) -> int:
    book_keeping_file = os.path.join(checkpoint_file, "BOOK_KEEPING_VARS.json")
    if not os.path.exists(book_keeping_file):
        print(f"BOOK_KEEPING_VARS.json not found in {checkpoint_file}!")
        return -1
    
    with open(book_keeping_file, "r") as f:
        book_keeping_vars = json.load(f)
    return book_keeping_vars["cumulative_timesteps"]

def get_most_recent_checkpoint() -> str:
    checkpoint_load_dir = r"data\checkpoints"
    if not os.path.exists(checkpoint_load_dir):
        return None
    
    # recursively search for every PPO_POLICY.pt file in the checkpoints folder
    checkpoint_files = []
    
    for root, _, files in os.walk(checkpoint_load_dir):
        for file in files:
            if file == "PPO_POLICY.pt":
                checkpoint_files.append(root)
    
    sorted_files = sorted(checkpoint_files, key=get_checkpoint_value)
    
    return sorted_files[-1]

def build_rocketsim_env():
    import rlgym_sim
    
    from rlgym_sim.utils import common_values

    from advanced_adapted_obs import AdvancedAdaptedObs
    from lookup_act import LookupAction
    
    from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition
    from rlgym_sim.utils.state_setters import RandomState

    state_setter = RandomState(True, True, False)

    from rewards.combined_rewards.hit_ball import HitBallReward

    rewards = HitBallReward()

    spawn_opponents = True
    team_size = 1
    tick_skip = 8

    no_touch_seconds = 10
    no_touch_ticks = int(round(no_touch_seconds * 120 / tick_skip))

    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_ticks)]

    reward_fn = rewards
    action_parser = LookupAction()
    obs_builder = AdvancedAdaptedObs(
            pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL, 
            player_padding=3, 
            expanding=False)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner

    n_proc = 40
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    ts_per_iteration = 50_000

    try:
        checkpoint_load_dir = get_most_recent_checkpoint()
        print(f"Loading checkpoint: {checkpoint_load_dir}")
    except:
        print("checkpoint load dir not found.")
        checkpoint_load_dir = None

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      checkpoint_load_folder=checkpoint_load_dir,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=ts_per_iteration,
                      ts_per_iteration=ts_per_iteration,
                      exp_buffer_size=ts_per_iteration*4,
                      ppo_minibatch_size=25_000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000,
                      policy_layer_sizes=[2048, 2048, 1024, 1024],
                      critic_layer_sizes=[2048, 2048, 1024, 1024],
                      timestep_limit=10e15,
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      render=True,
                      render_delay=8/120)
    
    start_time = time.time()

    learner.learn()

    end_time = time.time()
    trained_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(trained_time))
    
    print(f"Trained for {formatted_time}!")