import rlgym_sim
import os
import numpy as np
import torch
import rlgym_sim.utils.common_values as common_values

from tqdm import tqdm
from lookup_act import LookupAction
from advanced_adapted_obs import AdvancedAdaptedObs
from rlbot_implementation.discrete_policy import DiscreteFF
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

act = LookupAction()

OBS_SIZE = 231
POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]

num_actions = len(act._lookup_table)

class Agent:
    def __init__(self, checkpoint_path: str):
        PPO_PATH = os.path.join(checkpoint_path, "PPO_POLICY.pt")

        self.name = os.path.basename(checkpoint_path)
        self.policy = DiscreteFF(
            OBS_SIZE, num_actions, POLICY_LAYER_SIZES, torch.device("cuda:0")
        )
        self.policy.load_state_dict(
            torch.load(PPO_PATH, map_location="cuda", weights_only=True)
        )

    def get_action(self, state):
        with torch.no_grad():
            action_idx, _ = self.policy.get_action(state, True)
        return action_idx

    def __str__(self) -> str:
        return self.name


def match(agent1: Agent, agent2: Agent, num_games: int = 50, team_size: int = 2):
    agent1_score, agent2_score = 0, 0

    act = LookupAction()
    obs = AdvancedAdaptedObs(
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        player_padding=3,
        expanding=False,
    )
    term = [GoalScoredCondition()]

    with tqdm(total=num_games, desc=f"{agent1.name} vs {agent2.name}") as pbar:
        for _ in range(num_games):
            env = rlgym_sim.make(
                spawn_opponents=True,
                team_size=team_size,
                obs_builder=obs,
                action_parser=act,
                terminal_conditions=term,
            )

            state, info = env.reset(return_info=True)
            done = False

            while not done:
                actions = []
                for i, agent_state in enumerate(state):
                    action = None
                    if i < team_size:
                        action = agent1.get_action(agent_state)
                    else:
                        action = agent2.get_action(agent_state)
                    
                    actions.append(action)

                state, _, done, info = env.step(actions)

            result = info["result"]
            if result > 0:
                agent1_score += 1
            else:
                agent2_score += 1

            pbar.update(1)
            env.close()

        pbar.clear()
        pbar.close()

    winner = agent1 if agent1_score > agent2_score else agent2
    print(f"{winner.name} wins!")
    return [winner, agent1_score, agent2_score]


def load_all_agents(checkpoint_path: str):
    agents = []

    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            if file == "PPO_POLICY.pt":
                agents.append(Agent(root))

    return agents


if __name__ == "__main__":

    def tournament_iteration(agents: list[Agent], recursiveLimit: int = 100):
        matches = []
        next_agents = []

        if recursiveLimit == 0:
            return agents

        if len(agents) % 2 != 0:
            worst_agent = None
            least_cumulative_timesteps = float("inf")

            for i in range(len(agents)):
                agent = agents[i]
                cumulative_timesteps = int(agent.name)

                if cumulative_timesteps < least_cumulative_timesteps:
                    worst_agent = agent
                    least_cumulative_timesteps = cumulative_timesteps

            agents.remove(worst_agent)

        out = ""

        for i in range(len(agents) // 2):
            actualIndex = i * 2
            nextIndex = actualIndex + 1
            new_match = [agents[actualIndex], agents[nextIndex]]
            out = out + f"{new_match[0].name} vs {new_match[1].name}\n"
            matches.append(new_match)

        print(out)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(matches)):
                current_match = matches[i]
                futures.append(executor.submit(match, current_match[0], current_match[1]))

            for future in as_completed(futures):
                next_agents.append(future.result()[0])

        return tournament_iteration(next_agents, recursiveLimit - 1)

    agents = load_all_agents("data/checkpoints")
    recursiveLimit = int(np.log2(len(agents)))
    winner = tournament_iteration(agents, recursiveLimit)

    print(winner)
