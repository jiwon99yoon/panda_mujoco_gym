#/home/dyros/panda_mujoco_gym/panda_mujoco_gym/__init__.py
import os
from gymnasium.envs.registration import register

ENV_IDS = []

for task in ["Slide", "Push", "PickAndPlace"]:
    for reward_type in ["sparse", "dense"]:
        reward_suffix = "Dense" if reward_type == "dense" else "Sparse"
        env_id = f"Franka{task}{reward_suffix}-v0"

        register(
            id=env_id,
            entry_point=f"panda_mujoco_gym.envs:Franka{task}Env",
            kwargs={"reward_type": reward_type},
            max_episode_steps=100, # 1000은 너무 길고 50정도로 일단 
        )

        ENV_IDS.append(env_id)
