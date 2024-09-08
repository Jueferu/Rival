# Rival
Rival is a [Rocket League](https://www.rocketleague.com/en) bot trained for 3v3 soccer.
The bot is trained using [RLGym-PPO](https://github.com/AechPro/rlgym-ppo).

## How to play against it?
1. Install [RLBot](https://rlbot.org/)
2. Download Rival via `git clone https://github.com/AechPro/rlgym-ppo.git`
3. Execute RLBotGUI
4. Click the "+Add" button, then "Load Cfg File"
5. Choose `Rival\latest_bot.cfg`

Now you can put Rival against yourself or other bots!

(RIVAL CAN ONLY PLAY WITH 3 PLAYERS OR LESS PER TEAM!)

## Training Plan
At the end of each checkpoint, leave only the latest sub-checkpoint and delete the others, then commit with the "goal" of the training.

The training plan takes a lot of pages from [ZealanL's RLGym-PPO Guide](https://github.com/ZealanL/RLGym-PPO-Guide)

### STEP 1: Learning to touch the ball
```
ts_per_iteration = 50_000
learning_rate = 3e-4
ppo_epochs = 3
team_size = 1
state_setter = RandomState(True, True, False)
rewards = (
        (EventReward(touch=1), 20),
        (VelocityPlayerToBallReward(), 5),
        (PlayerFaceBallReward(), 1),
        (AirReward(), .15),
        )
```

Train for around 50k timesteps.
Training for this amount of timesteps is very overkill, but that's how I want it :)

### STEP 2: Learning to score
```
ts_per_iteration = 100_000
learning_rate = 2e-4
ppo_epochs = 3
team_size = 1
state_setter = (
        WallPracticeState(),
        KickoffLikeSetter(),
        GoaliePracticeState(),
        DribblingStateSetter(),
        JumpShotState(),
        SaveState(),
        SaveShot(),
        SideHighRoll(),
        ShotState(),
        RandomState(True, True, False),
    )
rewards = (
        (TouchBallRewardScaledByHitForce, 5),
        (VelocityPlayerToBallReward(), 2.5),
        (PlayerFaceBallReward(), .5),
        (AirReward(), .15),
        (VelocityBallToGoalReward(), 10),
        (EventReward(team_goal=1, concede=-0.5), 20),
    )
```

Train for around 100k timesteps