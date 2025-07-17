#/home/dyros/panda_mujoco_gym/train/common/wrappers.py
"""
환경 래퍼 모음
- RewardScalingWrapper: 보상 스케일링
- SuccessTrackingWrapper: 성공률 추적
"""

import gymnasium as gym
import numpy as np


class RewardScalingWrapper(gym.Wrapper):
    """보상 스케일링 래퍼"""
    
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale
        self.episode_reward = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        self.episode_reward += scaled_reward
        
        if terminated or truncated:
            info['original_reward'] = self.episode_reward / self.scale
            info['scaled_reward'] = self.episode_reward
            self.episode_reward = 0
            
        return obs, scaled_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.episode_reward = 0
        return self.env.reset(**kwargs)


class SuccessTrackingWrapper(gym.Wrapper):
    """성공률 추적 래퍼"""
    
    def __init__(self, env):
        super().__init__(env)
        self.success_count = 0
        self.episode_count = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            self.episode_count += 1
            if info.get('is_success', False):
                self.success_count += 1
            
            info['success_rate'] = self.success_count / self.episode_count if self.episode_count > 0 else 0
            info['total_episodes'] = self.episode_count
            info['total_successes'] = self.success_count
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
