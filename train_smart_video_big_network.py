#/home/minjun/panda_mujoco_gym/train_smart_video_big_network.py
#!/usr/bin/env python3
"""
스마트 비디오 녹화 시스템
- 에피소드 기반 녹화 (정지 화면 최소화)
- 1초 대기 후 자동 종료
- 6단계 시각화 (적절한 빈도)
- 의미 있는 trajectory만 캡처
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
import torch  # 🔧 torch 추가 (activation_fn 용)
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 사용자 정의 환경 등록
import panda_mujoco_gym

print("🤖 스마트 비디오 녹화 기반 Panda 로봇 학습!")
print("=" * 60)

# 보상 스케일링 래퍼 (기존과 동일)
class RewardScalingWrapper(gym.Wrapper):
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

# 성공률 추적 래퍼 (기존과 동일)
class SuccessTrackingWrapper(gym.Wrapper):
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
            
        return obs, reward, terminated, truncated, info

# 스마트 비디오 녹화 클래스
class SmartVideoRecorder:
    def __init__(self, env, video_path, max_episodes=5, wait_time=1.0):
        """
        스마트 비디오 녹화기
        
        Args:
            env: 환경
            video_path: 비디오 저장 경로
            max_episodes: 최대 에피소드 수
            wait_time: 에피소드 종료 후 대기 시간 (초)
        """
        self.env = env
        self.video_path = video_path
        self.max_episodes = max_episodes
        self.wait_time = wait_time
        self.wait_frames = int(wait_time * 50)  # 50 FPS 기준
        
        # 비디오 녹화 환경 설정
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
    def record_episodes(self, model=None, deterministic=True):
        """에피소드 기반 스마트 녹화"""
        print(f"🎥 스마트 비디오 녹화: 최대 {self.max_episodes} 에피소드")
        
        # 임시 비디오 환경
        video_env = DummyVecEnv([lambda: self.env])
        video_env = VecVideoRecorder(
            video_env,
            self.video_path,
            record_video_trigger=lambda x: x == 0,
            video_length=10000,  # 충분히 큰 값 (실제로는 에피소드로 제어)
            name_prefix="smart_recording"
        )
        
        obs = video_env.reset()
        episode_count = 0
        step_count = 0
        total_steps = 0
        successes = 0
        episode_rewards = []
        
        current_episode_reward = 0
        wait_counter = 0
        in_wait_phase = False
        
        while episode_count < self.max_episodes:
            if not in_wait_phase:
                # 정상 행동 단계
                if model is not None:
                    action, _ = model.predict(obs, deterministic=deterministic)
                else:
                    action = [video_env.action_space.sample()]
                
                obs, rewards, dones, infos = video_env.step(action)
                current_episode_reward += rewards[0]
                step_count += 1
                total_steps += 1
                
                if dones[0]:
                    # 에피소드 완료
                    episode_count += 1
                    success = infos[0].get('is_success', False)
                    if success:
                        successes += 1
                    
                    episode_rewards.append(current_episode_reward)
                    print(f"   Episode {episode_count}: {step_count} steps, "
                          f"reward: {current_episode_reward:.2f}, success: {success}")
                    
                    # 대기 단계 시작
                    in_wait_phase = True
                    wait_counter = 0
                    step_count = 0
                    current_episode_reward = 0
            
            else:
                # 대기 단계 (1초 대기)
                wait_counter += 1
                total_steps += 1
                
                # 대기 중에는 no-op action
                action = [np.zeros(video_env.action_space.shape[0])]
                obs, rewards, dones, infos = video_env.step(action)
                
                if wait_counter >= self.wait_frames:
                    # 대기 완료, 다음 에피소드 시작
                    in_wait_phase = False
                    if episode_count < self.max_episodes:
                        obs = video_env.reset()
        
        video_env.close()
        
        success_rate = successes / episode_count if episode_count > 0 else 0
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"✅ 스마트 비디오 완료!")
        print(f"   📊 총 에피소드: {episode_count}")
        print(f"   🏆 성공률: {success_rate:.3f}")
        print(f"   💎 평균 보상: {avg_reward:.2f}")
        print(f"   📹 총 프레임: {total_steps} (효율적!)")
        
        return {
            'episodes': episode_count,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'total_frames': total_steps
        }

# 스마트 설정 클래스
class SmartConfig:
    # 환경 설정
    env_name = "FrankaSlideDense-v0"
    algorithm = "SAC"
    total_timesteps = 1_000_000       # 더 많은 학습 유지
    
    # 🔥 FrankaSlideDense 특화 최적화된 하이퍼파라미터
    learning_rate = 1e-4              # 더 작은 학습률로 안정적 학습 (큰 네트워크에 적합)
    buffer_size = 1_000_000           # 더 큰 버퍼 (복잡한 학습에 필요)
    learning_starts = 10_000          # 더 많은 초기 경험 수집
    batch_size = 512                  # 더 큰 배치 (큰 네트워크에 적합)
    tau = 0.01                        # 더 빠른 타겟 네트워크 업데이트
    gamma = 0.98                      # 약간 낮은 할인 인수
    train_freq = 4                    # 더 자주 학습
    gradient_steps = 4                # 더 많은 gradient steps
    
    # 🧠 더 큰 네트워크 구조 (복잡한 로봇 태스크에 최적화)
    policy_kwargs = {
        "net_arch": [256, 256, 128],  # 🔥 추후 512, 512, 256으로 수정하기
        "activation_fn": torch.nn.ReLU,  # 🔧 문자열 → 함수 객체로 수정
        "normalize_images": False     # 이미지 정규화 비활성화
    }
    
    # 🎯 더 강한 탐험 노이즈 (큰 네트워크와 함께)
    action_noise_std = 0.2            # 더 큰 노이즈로 탐험 강화
    
    # 탐험 노이즈
    action_noise_std = 0.2
    
    # 평가 설정
    eval_freq = 20_000                # 더 자주 평가 (큰 네트워크 모니터링)
    n_eval_episodes = 20              # 더 많은 평가 에피소드
    eval_deterministic = True
    
    # 저장 설정
    save_freq = 100_000
    video_freq = 50_000               # 더 자주 비디오 생성
    
    # 🎥 스마트 비디오 설정
    max_episodes_per_video = 5        # 비디오당 최대 5 에피소드
    wait_time_after_episode = 1.0     # 에피소드 후 1초 대기
    
    # 시각화 설정
    enable_realtime_viz = True
    viz_duration = 45                 # 더 긴 시각화 시간 (큰 네트워크 관찰)
    
    # 환경 개선 설정
    normalize_env = True              # 관찰값 정규화 (큰 네트워크에 중요)
    reward_scale = 0.1                # 보상 스케일링 유지
    
    # 🧪 추가 학습 기법 (큰 네트워크 안정화)
    use_sde = True                    # State Dependent Exploration (큰 네트워크에 도움)
    sde_sample_freq = 4
    
    # 디렉토리 설정
    base_dir = "data_smart_video"
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "training_results")
    video_dir = os.path.join(base_dir, "videos")
    log_dir = os.path.join(base_dir, "logs")
    
    # 실험 이름
    experiment_name = f"{env_name}_{algorithm}_smart_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = SmartConfig()

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

class SmartTrainingCallback(BaseCallback):
    """
    스마트 학습 콜백 - 6단계 시각화 + 효율적 비디오
    """
    def __init__(self, log_dir, verbose=0):
        super(SmartTrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.csv_file = os.path.join(log_dir, f"training_log_{config.experiment_name}.csv")
        
        # 6단계 시각화 스케줄 (적절한 빈도)
        self.total_timesteps = config.total_timesteps
        self.visualization_steps = [
                                            # 0% - 학습 전 : train_model()에서 수동으로 생성됨
            self.total_timesteps // 5,      # 10% -> 20%
            self.total_timesteps * 2 // 5,       # 25% -> 40%
            self.total_timesteps * 3 // 5,       # 50% -> 60%
            self.total_timesteps * 4 // 5,   # 75% -> 80%
            self.total_timesteps              # 100% - 학습 완료
        ]
        self.completed_visualizations = set()
        
        # 성능 추적
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.recent_success_rate = 0.0
        
        # CSV 파일 초기화
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Episode', 'Reward', 'Length', 'Success', 'Success_Rate', 'Best_Reward'])
    
    def _on_step(self) -> bool:
        # 6단계 시각화 체크
        for i, target_step in enumerate(self.visualization_steps):
            if (target_step not in self.completed_visualizations and 
                self.num_timesteps >= target_step):
                
                self.completed_visualizations.add(target_step)
                stage_names = [
                    "1_학습진행_20퍼센트",
                    "2_학습진행_40퍼센트",
                    "3_학습진행_60퍼센트",
                    "4_학습진행_80퍼센트",
                    "5_학습완료_100퍼센트"
                ]
                stage_name = stage_names[i]
                
                print(f"\n🎬 [{stage_name}] 스마트 비디오 녹화! (Step {self.num_timesteps})")
                if self.recent_rewards:
                    print(f"   현재 평균 보상: {np.mean(self.recent_rewards[-50:]):.2f}")
                    print(f"   현재 성공률: {self.recent_success_rate:.3f}")
                self.record_smart_video(stage_name)
        
        # 에피소드 완료 시 처리
        if len(self.locals.get('dones', [])) > 0 and self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                is_success = info.get('is_success', False)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.recent_rewards.append(episode_reward)
                self.episode_count += 1
                
                if is_success:
                    self.success_count += 1
                
                # 성공률 계산
                self.recent_success_rate = self.success_count / self.episode_count if self.episode_count > 0 else 0
                
                # 최근 100개 에피소드만 유지
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)
                
                # 최고 보상 업데이트
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    print(f"🏆 새로운 최고 보상! {episode_reward:.2f} (성공률: {self.recent_success_rate:.3f})")
                
                # 로깅
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
                    avg_length = np.mean(self.episode_lengths[-50:]) if len(self.episode_lengths) >= 50 else np.mean(self.episode_lengths)
                    
                    print(f"📊 Episode {self.episode_count:4d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg: {avg_reward:7.2f} | "
                          f"Length: {episode_length:3.0f} | "
                          f"Success: {self.recent_success_rate:.3f}")
                
                # CSV 로그 저장
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self.episode_count,
                        episode_reward,
                        episode_length,
                        is_success,
                        self.recent_success_rate,
                        self.best_reward
                    ])
        
        return True
    
    def record_smart_video(self, stage_name):
        """스마트 비디오 녹화 - 에피소드 기반, 1초 대기"""
        try:
            print(f"🎥 [{stage_name}] 스마트 비디오 녹화 시작...")
            
            # 비디오 환경 생성
            video_env = create_env(config.env_name, render_mode="rgb_array")
            
            # 스마트 비디오 레코더 생성
            stage_video_dir = os.path.join(config.video_dir, "training_stages")
            video_path = os.path.join(stage_video_dir, f"{stage_name}_step_{self.num_timesteps}")
            
            recorder = SmartVideoRecorder(
                env=video_env,
                video_path=video_path,
                max_episodes=config.max_episodes_per_video,
                wait_time=config.wait_time_after_episode
            )
            
            # 스마트 녹화 실행
            stats = recorder.record_episodes(
                model=self.model if hasattr(self.model, 'predict') else None,
                deterministic=True
            )
            
            print(f"✅ [{stage_name}] 스마트 비디오 완료!")
            print(f"   📊 에피소드: {stats['episodes']}")
            print(f"   🏆 성공률: {stats['success_rate']:.3f}")
            print(f"   💎 평균 보상: {stats['avg_reward']:.2f}")
            print(f"   📹 효율적 프레임 수: {stats['total_frames']}")
            
        except Exception as e:
            print(f"⚠️ 스마트 비디오 녹화 오류: {e}")

def create_env(env_name, render_mode=None):
    """개선된 환경 생성 (래퍼 적용)"""
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)
    
    # 보상 스케일링 적용
    if config.reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=config.reward_scale)
    
    # 성공률 추적 추가
    env = SuccessTrackingWrapper(env)
    
    return env

def create_vec_env(env_name, n_envs=1, normalize=True):
    """벡터화된 환경 생성"""
    def make_env():
        env = create_env(env_name)
        return env
    
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    return vec_env

def create_model(env):
    """SAC 모델 생성"""
    n_actions = env.action_space.shape[-1]
    
    # 개선된 SAC 모델
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

def record_final_smart_video(model):
    """최종 스마트 비디오 녹화"""
    print("\n🎥 최종 성능 스마트 비디오 녹화...")
    
    # 최종 비디오용 환경
    video_env = create_env(config.env_name, render_mode="rgb_array")
    video_path = os.path.join(config.video_dir, f"final_performance_smart_{config.experiment_name}")
    
    # 더 많은 에피소드로 최종 성능 평가
    recorder = SmartVideoRecorder(
        env=video_env,
        video_path=video_path,
        max_episodes=10,  # 최종 평가는 더 많은 에피소드
        wait_time=1.0
    )
    
    stats = recorder.record_episodes(model=model, deterministic=True)
    
    print(f"🎥 최종 스마트 비디오 완료!")
    print(f"📊 총 에피소드: {stats['episodes']}")
    print(f"🏆 최종 성공률: {stats['success_rate']:.3f}")
    print(f"💎 최종 평균 보상: {stats['avg_reward']:.2f}")
    print(f"📹 효율적 총 프레임: {stats['total_frames']}")
    
    return stats

def train_model():
    """스마트 모델 학습"""
    print(f"🎯 환경: {config.env_name}")
    print(f"🧠 알고리즘: {config.algorithm}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
    print(f"🎥 스마트 비디오: 에피소드당 {config.max_episodes_per_video}개, {config.wait_time_after_episode}초 대기")
    print(f"🔧 보상 스케일링: {config.reward_scale}")
    print(f"📈 5단계 시각화 (20%, 40%, 60%, 80%, 100%) + 학습전 수동")
    print("-" * 60)
    
    create_directories()
    
    # 환경 생성
    print("🏗️  개선된 환경 생성 중...")
    env = create_vec_env(config.env_name, normalize=config.normalize_env)
    eval_env = create_vec_env(config.env_name, normalize=config.normalize_env)
    
    # 로거 설정
    logger_path = os.path.join(config.log_dir, config.experiment_name)
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])
    
    # 모델 생성
    print(f"\n🧠 {config.algorithm} 모델 초기화...")
    model = create_model(env)
    model.set_logger(new_logger)
    
    # 콜백 설정
    callbacks = []
    
    # 스마트 학습 모니터링
    training_callback = SmartTrainingCallback(config.log_dir)
    training_callback.model = model
    callbacks.append(training_callback)
    
    # 학습 전 스마트 비디오
    print("\n🎬 [학습 전] 무작위 행동 스마트 비디오!")
    training_callback.record_smart_video("0_학습전_무작위")
    
    # 평가 콜백
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
    
    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(config.model_dir, "checkpoints"),
        name_prefix=f"{config.algorithm}_smart_{config.experiment_name}"
    )
    callbacks.append(checkpoint_callback)
    
    # 학습 시작
    print(f"\n🚀 스마트 {config.algorithm} 학습 시작!")
    print("=" * 60)
    print("💡 6단계 자동 스마트 비디오!")
    print("💡 에피소드 기반 효율적 녹화!")
    print("💡 1초 대기 후 자동 전환!")
    print("💡 보상 스케일링으로 안정적 학습!")
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
    final_model_path = os.path.join(config.model_dir, f"{config.algorithm}_smart_{config.experiment_name}_final")
    model.save(final_model_path)
    print(f"💾 최종 모델 저장: {final_model_path}")
    
    # 최종 평가
    print("\n🔍 최종 평가...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, 
        n_eval_episodes=50,
        deterministic=True
    )
    print(f"🏆 최종 평가 결과: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model, training_callback

def main():
    """메인 실행 함수"""
    print(f"🚀 스마트 비디오 기반 Panda 로봇 학습 시작!")
    print(f"🎯 목표: FrankaSlideDense 태스크 마스터하기")
    print(f"💡 주요 특징:")
    print(f"   • 스마트 비디오: 에피소드 기반 효율적 녹화")
    print(f"   • 1초 대기: 정지 화면 최소화")
    print(f"   • 6단계 시각화: 적절한 모니터링")
    print(f"   • 보상 스케일링: {config.reward_scale} (안정적 학습)")
    print("=" * 60)
    
    # 학습 실행
    model, training_callback = train_model()
    
    # 최종 스마트 비디오 녹화
    final_stats = record_final_smart_video(model)
    
    print("\n" + "=" * 60)
    print("🎉 스마트 학습 완료!")
    print(f"📊 최종 성공률: {training_callback.recent_success_rate:.3f}")
    print(f"🏆 최고 보상: {training_callback.best_reward:.2f}")
    print(f"🎥 효율적 비디오 시스템 적용 완료!")
    print("=" * 60)
    
    return model, training_callback, final_stats

if __name__ == "__main__":
    model, training_callback, final_stats = main()