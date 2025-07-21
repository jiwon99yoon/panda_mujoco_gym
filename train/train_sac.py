#/home/dyros/panda_mujoco_gym/train/train_sac.py
#!/usr/bin/env python3
"""
SAC 학습 스크립트
- 순수하게 학습에만 집중
- 결과는 outputs 폴더에 저장
- 비디오 녹화는 별도 스크립트에서 수행
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 사용자 정의 환경 등록
import panda_mujoco_gym

# Common 모듈 임포트
from train.common.config   import SACConfig
from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper
from train.common.callbacks import TrainingCallback
#from common import SACConfig, RewardScalingWrapper, SuccessTrackingWrapper, TrainingCallback

# 한 에피소드 당 step을 기본 50 -> 1,000으로 wrapping하기 위해
from gymnasium.wrappers import TimeLimit

def create_env(env_name, render_mode=None, reward_scale=1.0):
    """환경 생성 (래퍼 적용)"""
    raw = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(raw, max_episode_steps = 1000) #episode의 timestep 1000으로 할당
    env = Monitor(env)
    
    # 보상 스케일링 적용
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)
    
    # 성공률 추적 추가
    env = SuccessTrackingWrapper(env)
    
    return env


# 각 워커에 서로 다른 시드를 할당
def create_vec_env(env_name, n_envs=1, normalize=True, reward_scale=1.0, seed=None):
    def make_env(rank):
        def _init():
            env = create_env(env_name, reward_scale=reward_scale)

            # 수정: seed 설정 제거 (매번 랜덤하게)
            # seed가 있으면 action/observation space의 샘플링용으로만 사용
            if seed is not None:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env
        return _init

    # n_envs가 1이면 디버깅 용이한 DummyVecEnv 사용
    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method='fork')    
    
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
 
    return vec_env
# def create_vec_env(env_name, n_envs=1, normalize=True, reward_scale=1.0):
#     """벡터화된 환경 생성"""
#     def make_env():
#         return create_env(env_name, reward_scale=reward_scale)
    
#     #vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
#     # SubprocVecEnv: 서로 다른 프로세스에서 병렬 시뮬레이션 실행
#     vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    
#     if normalize:
#         vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
#     return vec_env


def create_sac_model(env, config):
    """SAC 모델 생성"""
    n_actions = env.action_space.shape[-1]
    
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=config.action_noise_std * np.ones(n_actions)
    )
    
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        action_noise=action_noise,
        policy_kwargs=config.policy_kwargs,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        verbose=1,
        tensorboard_log=config.log_dir,
        device="auto"
    )
    
    return model


def train_sac(config: SACConfig):
    """SAC 학습 메인 함수"""
    print("🚀 SAC 학습 시작!")
    print(f"🎯 환경: {config.env_name}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
    print(f"💾 결과 저장 위치: {config.exp_dir}")
    print("-" * 60)
    
    # 디렉토리 생성
    config.create_directories()
    
    # 환경 생성
    print("🏗️  환경 생성 중...")
    env = create_vec_env(
        config.env_name,
        n_envs=config.n_envs, 
        normalize=config.normalize_env, 
        reward_scale=config.reward_scale,
        seed=config.seed
    )
    eval_env = create_vec_env(
        config.env_name,
        n_envs=1,  #단일환경에서 평가하기
        normalize=config.normalize_env,
        reward_scale=config.reward_scale,
        seed=config.seed
    )
    
    # 로거 설정
    logger_path = os.path.join(config.log_dir, "tensorboard")
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])
    
    # 모델 생성
    print(f"\n🧠 SAC 모델 초기화...")
    model = create_sac_model(env, config)
    model.set_logger(new_logger)
    
    # 콜백 설정
    callbacks = []
    
    # 학습 모니터링 콜백
    training_callback = TrainingCallback(config)
    callbacks.append(training_callback)
    
    # 평가 콜백
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.model_dir, "best_model"),
        log_path=os.path.join(config.log_dir, "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.eval_deterministic,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.checkpoint_dir,
        name_prefix=f"sac_{config.env_name}",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # 학습 시작
    print(f"\n🚀 학습 시작!")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  학습이 중단되었습니다.")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ 학습 완료!")
    print(f"⏱️  총 학습 시간: {training_time/3600:.2f}시간")
    
    # 최종 모델 저장
    final_model_path = os.path.join(config.model_dir, "final_model")
    model.save(final_model_path)
    env.save(os.path.join(config.model_dir, "vec_normalize.pkl"))
    print(f"💾 최종 모델 저장: {final_model_path}")
    
    # 최종 평가
    print("\n🔍 최종 평가...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, 
        n_eval_episodes=50, #n_eval_episode : 평가 시뮬레이션 횟수(개수)
        deterministic=True
    )
    print(f"🏆 최종 평가 결과: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 학습 요약 저장
    import json
    # config 안에 JSON 직렬화 불가 객체를 문자열로 변환
    serializable_config = {}
    for k, v in config.__dict__.items():
        try:
            json.dumps({k: v})
            serializable_config[k] = v
        except TypeError:
            serializable_config[k] = repr(v)
    summary = {
        "experiment_name": config.experiment_name,
        "env_name": config.env_name,
        "algorithm": "SAC",
        "total_timesteps": config.total_timesteps,
        "training_time_hours": training_time / 3600,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "best_reward": float(training_callback.best_reward),
        "final_success_rate": float(training_callback.recent_success_rate),
        "total_episodes": training_callback.episode_count,
        "config": serializable_config
    }
    
    summary_path = os.path.join(config.exp_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n📋 학습 요약 저장: {summary_path}")
    
    return model, env


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SAC 학습 스크립트")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0", help="환경 이름")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="총 학습 스텝")
    parser.add_argument("--exp-name", type=str, default=None, help="실험 이름")
    parser.add_argument("--reward-scale", type=float, default=0.1, help="보상 스케일")
    #main()에 n_envs 인자 추가
    parser.add_argument("--n-envs", type=int, default=4, help="병렬 환경 개수")
    #seed 인자 추가
    parser.add_argument("--seed", type=int, default=None, help="난수 시드 (worker마다 seed+rank 적용)")     

    args = parser.parse_args()
    
    # 설정 생성
    config = SACConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        experiment_name=args.exp_name,
        reward_scale=args.reward_scale,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    
    # 학습 실행
    model, env = train_sac(config)
    
    print("\n" + "=" * 60)
    print("🎉 학습 완료!")
    print(f"📁 결과 저장 위치: {config.exp_dir}")
    print("📋 다음 단계:")
    print(f"   python evaluate/evaluate_with_video.py --exp-dir {config.exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
