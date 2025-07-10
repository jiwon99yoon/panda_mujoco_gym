#!/usr/bin/env python3
"""
Stable-Baselines3 SAC 기반 Panda 로봇 학습 코드
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 사용자 정의 환경 등록
import panda_mujoco_gym

print("🤖 Stable-Baselines3 SAC 기반 Panda 로봇 학습!")
print("=" * 60)

# 설정 클래스
class Config:
    # 환경 설정
    env_name = "FrankaSlideDense-v0"  # 학습할 환경
    total_timesteps = 200_000         # 총 학습 스텝
    
    # SAC 하이퍼파라미터
    learning_rate = 3e-4
    buffer_size = 1_000_000
    learning_starts = 1000
    batch_size = 256
    tau = 0.005
    gamma = 0.99
    train_freq = 1
    gradient_steps = 1
    
    # 평가 설정
    eval_freq = 10_000
    n_eval_episodes = 10
    eval_deterministic = True
    
    # 저장 설정
    save_freq = 50_000
    video_freq = 50_000
    video_length = 1000
    
    # 디렉토리 설정
    base_dir = "data"
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "training_results")
    video_dir = os.path.join(base_dir, "videos")
    log_dir = os.path.join(base_dir, "logs")
    
    # 실험 이름 (타임스탬프 포함)
    experiment_name = f"{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = Config()

# 디렉토리 생성
def create_directories():
    """필요한 디렉토리들을 생성합니다."""
    dirs_to_create = [
        config.base_dir,
        config.model_dir,
        config.results_dir,
        config.video_dir,
        config.log_dir
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 디렉토리 생성: {dir_path}")

# 커스텀 콜백 클래스
class TrainingCallback(BaseCallback):
    """
    학습 과정을 모니터링하고 로그를 저장하는 콜백
    """
    def __init__(self, log_dir, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.csv_file = os.path.join(log_dir, f"training_log_{config.experiment_name}.csv")
        
        # CSV 파일 초기화
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Episode', 'Reward', 'Length', 'FPS'])
    
    def _on_step(self) -> bool:
        # 에피소드가 끝났을 때
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # 콘솔 출력
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"📊 Episode {len(self.episode_rewards):4d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg Reward: {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d}")
                
                # CSV 로그 저장
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    fps = self.locals.get('fps', 0)
                    writer.writerow([
                        self.num_timesteps,
                        len(self.episode_rewards),
                        episode_reward,
                        episode_length,
                        fps
                    ])
        
        return True

def create_env(env_name, render_mode=None):
    """환경을 생성합니다."""
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)
    return env

def train_model():
    """SAC 모델을 학습합니다."""
    print(f"🎯 환경: {config.env_name}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
    print(f"🧠 알고리즘: SAC (Stable-Baselines3)")
    print("-" * 60)
    
    # 디렉토리 생성
    create_directories()
    
    # 환경 생성
    print("🏗️  환경 생성 중...")
    env = create_env(config.env_name)
    eval_env = create_env(config.env_name)
    
    # 환경 정보 출력
    print(f"📏 관찰 공간: {env.observation_space}")
    print(f"🎮 행동 공간: {env.action_space}")
    print(f"🎯 행동 범위: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")
    
    # 로거 설정
    logger_path = os.path.join(config.log_dir, config.experiment_name)
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])
    
    # SAC 모델 생성
    print("\n🧠 SAC 모델 초기화...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        verbose=1,
        tensorboard_log=config.log_dir,
        device="auto"  # GPU 자동 감지
    )
    
    # 로거 설정
    model.set_logger(new_logger)
    
    # 콜백 설정
    callbacks = []
    
    # 1. 학습 모니터링 콜백
    training_callback = TrainingCallback(config.log_dir)
    callbacks.append(training_callback)
    
    # 2. 평가 콜백
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.model_dir, f"best_model_{config.experiment_name}"),
        log_path=os.path.join(config.log_dir, "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.eval_deterministic,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # 3. 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(config.model_dir, "checkpoints"),
        name_prefix=f"sac_{config.experiment_name}"
    )
    callbacks.append(checkpoint_callback)
    
    # 학습 시작
    print("\n🚀 학습 시작!")
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
    final_model_path = os.path.join(config.model_dir, f"sac_{config.experiment_name}_final")
    model.save(final_model_path)
    print(f"💾 최종 모델 저장: {final_model_path}")
    
    # 최종 평가
    print("\n🔍 최종 평가 진행...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, 
        n_eval_episodes=20, 
        deterministic=True
    )
    print(f"🏆 최종 평가 결과: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model, training_callback

def record_videos(model):
    """학습된 모델의 동영상을 녹화합니다."""
    print("\n🎥 동영상 녹화 시작...")
    
    # 비디오 녹화용 환경 설정
    video_env = create_env(config.env_name, render_mode="rgb_array")
    video_env = DummyVecEnv([lambda: video_env])
    
    video_path = os.path.join(config.video_dir, f"final_performance_{config.experiment_name}")
    video_env = VecVideoRecorder(
        video_env,
        video_path,
        record_video_trigger=lambda x: x == 0,
        video_length=config.video_length,
        name_prefix=f"sac_{config.experiment_name}"
    )
    
    # 동영상 녹화
    obs = video_env.reset()
    for i in range(config.video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = video_env.step(action)
        if dones[0]:
            obs = video_env.reset()
    
    video_env.close()
    print(f"🎥 동영상 저장 완료: {video_path}")

def save_training_stats(training_callback):
    """학습 통계를 저장합니다."""
    print("\n📊 학습 통계 저장...")
    
    stats = {
        'experiment_name': config.experiment_name,
        'env_name': config.env_name,
        'total_timesteps': config.total_timesteps,
        'total_episodes': len(training_callback.episode_rewards),
        'final_avg_reward': np.mean(training_callback.episode_rewards[-100:]) if len(training_callback.episode_rewards) >= 100 else np.mean(training_callback.episode_rewards),
        'best_reward': np.max(training_callback.episode_rewards) if training_callback.episode_rewards else 0,
        'config': {
            'learning_rate': config.learning_rate,
            'buffer_size': config.buffer_size,
            'batch_size': config.batch_size,
            'tau': config.tau,
            'gamma': config.gamma
        },
        'episode_rewards': training_callback.episode_rewards,
        'episode_lengths': training_callback.episode_lengths
    }
    
    # JSON으로 저장
    stats_path = os.path.join(config.results_dir, f"training_stats_{config.experiment_name}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📋 통계 저장 완료: {stats_path}")
    return stats

def plot_training_results(stats):
    """학습 결과를 시각화합니다."""
    print("\n📈 학습 결과 시각화...")
    
    episode_rewards = stats['episode_rewards']
    episode_lengths = stats['episode_lengths']
    
    if not episode_rewards:
        print("⚠️  시각화할 데이터가 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Results - {config.experiment_name}', fontsize=16)
    
    # 1. 에피소드별 보상
    axes[0, 0].plot(episode_rewards, alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # 2. 이동 평균 보상
    if len(episode_rewards) > 50:
        window = min(50, len(episode_rewards) // 4)
        moving_avg = []
        for i in range(window, len(episode_rewards)):
            moving_avg.append(np.mean(episode_rewards[i-window:i]))
        
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
    
    # 3. 에피소드 길이
    axes[0, 2].plot(episode_lengths, color='orange', alpha=0.7)
    axes[0, 2].set_title('Episode Lengths')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True)
    
    # 4. 보상 분포
    axes[1, 0].hist(episode_rewards, bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # 5. 길이 분포
    axes[1, 1].hist(episode_lengths, bins=50, alpha=0.7, color='red')
    axes[1, 1].set_title('Episode Length Distribution')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    # 6. 학습 진행률
    if len(episode_rewards) > 100:
        progress_rewards = []
        chunk_size = len(episode_rewards) // 10
        for i in range(0, len(episode_rewards), chunk_size):
            chunk = episode_rewards[i:i+chunk_size]
            if chunk:
                progress_rewards.append(np.mean(chunk))
        
        axes[1, 2].plot(progress_rewards, marker='o')
        axes[1, 2].set_title('Learning Progress (10 chunks)')
        axes[1, 2].set_xlabel('Training Phase')
        axes[1, 2].set_ylabel('Average Reward')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # 그래프 저장
    plot_path = os.path.join(config.results_dir, f'training_results_{config.experiment_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 그래프 저장 완료: {plot_path}")

def main():
    """메인 함수"""
    print("🚀 Stable-Baselines3 SAC 학습 파이프라인 시작!")
    
    try:
        # 1. 모델 학습
        model, training_callback = train_model()
        
        # 2. 동영상 녹화
        record_videos(model)
        
        # 3. 통계 저장
        stats = save_training_stats(training_callback)
        
        # 4. 결과 시각화
        plot_training_results(stats)
        
        # 5. 결과 요약
        print("\n" + "="*60)
        print("🎉 모든 작업 완료!")
        print("="*60)
        print(f"📁 저장된 파일들:")
        print(f"   🧠 모델: {config.model_dir}/")
        print(f"   📊 결과: {config.results_dir}/")
        print(f"   🎥 동영상: {config.video_dir}/")
        print(f"   📋 로그: {config.log_dir}/")
        print(f"   📈 TensorBoard: tensorboard --logdir {config.log_dir}")
        
        if training_callback.episode_rewards:
            print(f"\n📊 최종 통계:")
            print(f"   🎯 총 에피소드: {len(training_callback.episode_rewards)}")
            print(f"   🏆 최고 보상: {np.max(training_callback.episode_rewards):.2f}")
            print(f"   📈 최종 평균: {np.mean(training_callback.episode_rewards[-100:]):.2f}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
