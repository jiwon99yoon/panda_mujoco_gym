#/home/minjun/panda_mujoco_gym/train_sac_her.py
import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

env = gym.make("FrankaPickAndPlaceSparse-v0")
env = Monitor(env)  # 모니터로 감싸기

model = SAC(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",  # 'future', 'final', 'episode'
    ),
    verbose=1,
    tensorboard_log="./logs/sac_her/"
)

model.learn(total_timesteps=100_000)
model.save("sac_her_pick_and_place")

