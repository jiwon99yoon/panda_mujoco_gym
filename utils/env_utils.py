# /home/dyros/panda_mujoco_gym/utils/env_utils.py

"""
환경 생성 유틸리티 함수들
"""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# train/common에서 래퍼 임포트
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def create_env(env_name, render_mode=None, reward_scale=1.0, add_wrappers=True):
    """환경 생성 (래퍼 적용 옵션)"""
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)
    
    if add_wrappers:
        # 보상 스케일링 적용
        if reward_scale != 1.0:
            env = RewardScalingWrapper(env, scale=reward_scale)
        
        # 성공률 추적 추가
        env = SuccessTrackingWrapper(env)
    
    return env


def create_vec_env(env_name, n_envs=1, normalize=True, reward_scale=1.0, 
                   vec_normalize_path=None, training=True, render_mode=None):
    """
    벡터화된 환경 생성
    
    Args:
        env_name: 환경 이름
        n_envs: 환경 개수
        normalize: 정규화 여부
        reward_scale: 보상 스케일
        vec_normalize_path: 기존 정규화 통계 파일 경로
        training: 학습 모드 여부 (False면 평가 모드)
    """
    def make_env():
        return create_env(env_name, render_mode=render_mode, reward_scale=reward_scale)
    
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    if normalize:
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            # 기존 통계 로드
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            vec_env.training = training
        else:
            # 새로 생성
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    return vec_env
