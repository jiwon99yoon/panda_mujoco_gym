# #/home/minjun/panda_mujoco_gym/train_sac_highlight.py
# # /home/dyros/panda_mujoco_gym/train_sac_highlight.py

#!/usr/bin/env python3
"""
🎯 완전히 수정된 에피소드 기반 비디오 녹화 시스템
- 학습 환경과 완전히 일치하는 비디오 환경 (VecNormalize 포함)
- 실제 info['is_success'] 기준 사용
- Eval 환경과 동일한 조건으로 비디오 제작
- Best Model 기반 비교 비디오
- 성공 에피소드 우선 녹화
"""

import os
import sys
import time
import numpy as np
import cv2
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
import pickle

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 사용자 정의 환경 등록
import panda_mujoco_gym

print("🎯 완전히 수정된 에피소드 기반 비디오 녹화 시스템!")
print("=" * 60)

# 보상 스케일링 래퍼
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

# 성공률 추적 래퍼
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

# 🎥 개선된 에피소드별 비디오 녹화 클래스
class FixedEpisodeVideoRecorder:
    """
    수정된 에피소드별 개별 비디오 녹화 시스템
    - VecNormalize 상태 동기화
    - 실제 info['is_success'] 기준 사용
    - 성공 에피소드 우선 녹화
    """
    
    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = save_dir
        self.fps = fps
        self.current_episode_frames = []
        self.episode_metadata = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start_episode_recording(self):
        """새 에피소드 녹화 시작"""
        self.current_episode_frames = []
        
    def add_frame(self, frame: np.ndarray):
        """현재 에피소드에 프레임 추가"""
        if frame is not None:
            self.current_episode_frames.append(frame.copy())
    
    def end_episode_recording(self, episode_info: Dict) -> Optional[str]:
        """
        에피소드 녹화 종료 및 비디오 저장
        
        Args:
            episode_info: {
                'episode_id': int,
                'reward': float, 
                'length': int,
                'success': bool,  # ✅ 실제 info['is_success'] 사용
                'stage': str
            }
        """
        if not self.current_episode_frames:
            return None
            
        # 파일명 생성
        stage = episode_info.get('stage', 'unknown')
        episode_id = episode_info.get('episode_id', 0)
        success_str = 'SUCCESS' if episode_info.get('success', False) else 'FAIL'
        reward = episode_info.get('reward', 0)
        
        filename = f"{stage}_ep{episode_id:03d}_{success_str}_reward{reward:.1f}.mp4"
        video_path = os.path.join(self.save_dir, filename)
        
        # 비디오 저장
        success = self._save_video(video_path, self.current_episode_frames)
        
        if success:
            # 메타데이터 저장
            episode_info['video_path'] = video_path
            episode_info['frame_count'] = len(self.current_episode_frames)
            self.episode_metadata.append(episode_info)
            
            print(f"✅ 에피소드 비디오 저장: {filename}")
            print(f"   프레임 수: {len(self.current_episode_frames)}, 성공: {success_str}, 보상: {reward:.2f}")
            
            return video_path
        
        return None
    
    def _save_video(self, video_path: str, frames: List[np.ndarray]) -> bool:
        """프레임들을 비디오 파일로 저장"""
        if not frames:
            return False
            
        try:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            for frame in frames:
                # RGB → BGR 변환 (OpenCV용)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
            
            writer.release()
            return True
            
        except Exception as e:
            print(f"⚠️ 비디오 저장 오류: {e}")
            return False

# 🎯 수정된 학습 진행률별 비디오 시스템
class FixedStageBasedVideoSystem:
    """
    수정된 학습 진행률별 비디오 시스템
    - 학습 환경과 완전히 일치하는 비디오 환경
    - VecNormalize 상태 동기화
    - 실제 성공 기준 사용
    - 성공 에피소드 우선 녹화
    """
    
    def __init__(self, base_dir: str, total_timesteps: int):
        self.base_dir = base_dir
        self.total_timesteps = total_timesteps
        
        # 6단계 정의
        self.stages = {
            '0_random': 0,                           # 학습 전
            '1_20percent': total_timesteps // 5,     # 20%
            '2_40percent': total_timesteps * 2 // 5, # 40%
            '3_60percent': total_timesteps * 3 // 5, # 60% 
            '4_80percent': total_timesteps * 4 // 5, # 80%
            '5_100percent': total_timesteps          # 100%
        }
        
        self.completed_stages = set()
        self.stage_videos = {}  # stage_name -> [video_paths]
        self.best_episodes = []  # 최고 보상 에피소드들
        
        # 각 단계별 디렉토리 생성
        for stage_name in self.stages.keys():
            stage_dir = os.path.join(base_dir, stage_name)
            os.makedirs(stage_dir, exist_ok=True)
            self.stage_videos[stage_name] = []
    
    def should_record_stage(self, current_timestep: int) -> Optional[str]:
        """현재 timestep에서 녹화할 단계가 있는지 확인"""
        for stage_name, target_step in self.stages.items():
            if (stage_name not in self.completed_stages and 
                current_timestep >= target_step):
                return stage_name
        return None
    
    def record_stage_episodes(self, stage_name: str, model, eval_env_stats, num_episodes: int = 3):
        """
        🎯 수정된 에피소드 녹화 - Eval 환경과 완전히 일치
        
        Args:
            stage_name: 단계 이름
            model: 학습된 모델
            eval_env_stats: Eval 환경의 VecNormalize 통계
            num_episodes: 녹화할 에피소드 수
        """
        print(f"\n🎬 [{stage_name}] 에피소드 녹화 시작! (Eval 환경 기준)")
        
        stage_dir = os.path.join(self.base_dir, stage_name)
        recorder = FixedEpisodeVideoRecorder(stage_dir)
        
        # 🔧 Eval 환경과 동일한 비디오 환경 생성
        video_env = self._create_eval_identical_env(eval_env_stats)
        
        episode_results = []
        attempts = 0
        max_attempts = num_episodes * 10  # 성공 에피소드를 위해 더 많은 시도
        
        for episode_idx in range(num_episodes):
            print(f"  에피소드 {episode_idx + 1}/{num_episodes} 녹화 중...")
            
            # 🎯 성공 에피소드 우선 녹화
            episode_found = False
            while attempts < max_attempts and not episode_found:
                attempts += 1
                result = self._record_single_episode(
                    recorder, model, video_env, episode_idx, stage_name
                )
                
                if result:
                    episode_results.append(result)
                    self.stage_videos[stage_name].append(result['video_path'])
                    
                    # 최고 보상 추적
                    if not self.best_episodes or result['reward'] > min(ep['reward'] for ep in self.best_episodes):
                        self.best_episodes.append(result)
                        self.best_episodes.sort(key=lambda x: x['reward'], reverse=True)
                        self.best_episodes = self.best_episodes[:10]  # 상위 10개만 유지
                    
                    episode_found = True  # 에피소드 완료
                
                # 성공 에피소드를 찾기 위해 더 많이 시도
                if attempts % 5 == 0:
                    print(f"    시도 {attempts}: 성공 에피소드 탐색 중...")
        
        self.completed_stages.add(stage_name)
        print(f"✅ [{stage_name}] 완료! {len(episode_results)}개 에피소드 녹화됨")
        
        return episode_results
    
    def _create_eval_identical_env(self, eval_env_stats):
        """
        🎯 Eval 환경과 완전히 동일한 비디오 환경 생성
        """
        # 기본 환경 생성 (학습 환경과 동일한 래퍼들 적용)
        env = gym.make(config.env_name, render_mode="rgb_array")
        env = Monitor(env)
        
        # 보상 스케일링 적용
        if config.reward_scale != 1.0:
            env = RewardScalingWrapper(env, scale=config.reward_scale)
        
        # 성공률 추적 추가
        env = SuccessTrackingWrapper(env)
        
        # 벡터화
        vec_env = DummyVecEnv([lambda: env])
        
        # 🔧 VecNormalize 적용 및 통계 동기화
        if config.normalize_env and eval_env_stats is not None:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
            # Eval 환경의 정규화 통계 복사
            vec_env.obs_rms = eval_env_stats['obs_rms']
            vec_env.ret_rms = eval_env_stats['ret_rms']
            vec_env.training = False  # 평가 모드
        
        return vec_env
    
    def _record_single_episode(self, recorder, model, video_env, episode_idx, stage_name):
        """단일 에피소드 녹화"""
        recorder.start_episode_recording()
        
        obs = video_env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        # 초기 프레임 추가 (올바른 전체 이미지)
        rendered = video_env.render()
        if rendered is not None:
            img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
            recorder.add_frame(img)

        while not done:
            # 액션 예측 (deterministic=True로 Eval과 동일)
            if model and hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = video_env.action_space.sample()  # 랜덤 액션
            
            # 환경 스텝
            obs, rewards, dones, infos = video_env.step(action)
            done = dones[0]
            total_reward += rewards[0]
            step_count += 1
            
            # 프레임 추가 (올바른 전체 이미지)
            rendered = video_env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                recorder.add_frame(img)
            
            # 무한 루프 방지
            if step_count > 1000:
                break
        
        # 🎯 실제 성공 기준 사용
        episode_info = {
            'episode_id': episode_idx,
            'reward': total_reward,
            'length': step_count,
            'success': infos[0].get('is_success', False),  # ✅ 실제 성공 기준
            'stage': stage_name
        }
        
        # 비디오 저장
        video_path = recorder.end_episode_recording(episode_info)
        
        if video_path:
            episode_info['video_path'] = video_path
            return episode_info
        
        return None
    
    def create_highlight_video(self):
        """최고 보상 에피소드들로 하이라이트 비디오 생성"""
        if not self.best_episodes:
            print("❌ 하이라이트 비디오 생성 실패: 에피소드 없음")
            return
        
        print(f"\n🌟 하이라이트 비디오 생성 중... (상위 {len(self.best_episodes)}개 에피소드)")
        
        highlight_dir = os.path.join(self.base_dir, "highlights")
        os.makedirs(highlight_dir, exist_ok=True)
        
        # 각 최고 에피소드를 하이라이트 폴더에 복사
        import shutil
        for i, episode in enumerate(self.best_episodes):
            original_path = episode['video_path']
            if os.path.exists(original_path):
                highlight_filename = f"highlight_{i+1:02d}_reward{episode['reward']:.1f}_{episode['stage']}.mp4"
                highlight_path = os.path.join(highlight_dir, highlight_filename)
                
                # 파일 복사
                shutil.copy2(original_path, highlight_path)
                print(f"  ✨ {highlight_filename}")
        
        print(f"✅ 하이라이트 비디오 완료! {highlight_dir}")

# 설정 클래스
class FixedConfig:
    # 환경 설정
    env_name = "FrankaSlideDense-v0"
    algorithm = "SAC"
    total_timesteps = 1_000_000
    
    # SAC 하이퍼파라미터
    learning_rate = 1e-4
    buffer_size = 1_000_000
    learning_starts = 10_000
    batch_size = 512
    tau = 0.01
    gamma = 0.98
    train_freq = 4
    gradient_steps = 4
    
    # 네트워크 구조
    policy_kwargs = {
        "net_arch": [256, 256, 128],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False
    }
    
    # 탐험 노이즈
    action_noise_std = 0.2
    
    # 평가 설정
    eval_freq = 20_000
    n_eval_episodes = 20
    eval_deterministic = True
    
    # 저장 설정
    save_freq = 100_000
    
    # 🎥 수정된 비디오 설정
    episodes_per_stage = 3         # 각 단계별 에피소드 수
    video_fps = 30                 # 비디오 FPS
    
    # 환경 개선 설정
    normalize_env = True
    reward_scale = 0.1
    
    # 추가 학습 기법
    use_sde = True
    sde_sample_freq = 4
    
    # 디렉토리 설정
    base_dir = "fixed_episode_videos"
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "training_results")
    video_dir = os.path.join(base_dir, "videos")
    log_dir = os.path.join(base_dir, "logs")
    
    # 실험 이름
    experiment_name = f"{env_name}_{algorithm}_fixed_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = FixedConfig()

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

# 🎯 수정된 학습 콜백
class FixedTrainingCallback(BaseCallback):
    """
    수정된 학습 콜백 - Eval 환경과 일치하는 비디오 시스템
    """
    def __init__(self, log_dir, verbose=0):
        super(FixedTrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.csv_file = os.path.join(log_dir, f"training_log_{config.experiment_name}.csv")
        
        # 수정된 비디오 시스템 초기화
        self.video_system = FixedStageBasedVideoSystem(
            base_dir=config.video_dir,
            total_timesteps=config.total_timesteps
        )
        
        # 성능 추적
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.recent_success_rate = 0.0
        
        # 🔧 Eval 환경 통계 저장용
        self.eval_env_stats = None
        
        # CSV 파일 초기화
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Episode', 'Reward', 'Length', 'Success', 'Success_Rate', 'Best_Reward'])
    
    def set_eval_env_stats(self, eval_env):
        """Eval 환경의 정규화 통계 저장"""
        if hasattr(eval_env, 'obs_rms') and hasattr(eval_env, 'ret_rms'):
            self.eval_env_stats = {
                'obs_rms': eval_env.obs_rms,
                'ret_rms': eval_env.ret_rms
            }
            #print("✅ Eval 환경 통계 저장 완료!")
    
    def _on_step(self) -> bool:
        # 단계별 비디오 녹화 체크
        stage_to_record = self.video_system.should_record_stage(self.num_timesteps)
        if stage_to_record:
            print(f"\n🎬 [{stage_to_record}] 수정된 비디오 녹화! (Step {self.num_timesteps})")
            if self.recent_rewards:
                print(f"   현재 평균 보상: {np.mean(self.recent_rewards[-50:]):.2f}")
                print(f"   현재 성공률: {self.recent_success_rate:.3f}")
            
            self.record_stage_videos(stage_to_record)
        
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
    
    def record_stage_videos(self, stage_name):
        """🎯 수정된 단계별 비디오 녹화"""
        try:
            # 🔧 Eval 환경 통계와 함께 비디오 녹화
            results = self.video_system.record_stage_episodes(
                stage_name=stage_name,
                model=self.model if hasattr(self, 'model') else None,
                eval_env_stats=self.eval_env_stats,
                num_episodes=config.episodes_per_stage
            )
            
            print(f"✅ [{stage_name}] 수정된 비디오 녹화 완료!")
            print(f"   📊 녹화된 에피소드: {len(results)}개")
            if results:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                avg_reward = np.mean([r['reward'] for r in results])
                print(f"   🏆 성공률: {success_rate:.3f}")
                print(f"   💎 평균 보상: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"⚠️ 비디오 녹화 오류: {e}")

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

def fixed_compare_best_vs_final_models(final_model, training_callback):
    print("\n🔍 시점별 Best Model vs Final Model 성능 비교...")

    comparison_results = {}

    # 1) model_dir 에서 best_model_* 폴더만 골라서 순회
    for sub in sorted(os.listdir(config.model_dir)):
        if not sub.startswith("best_model_"):
            continue
        stage = sub.replace("best_model_", "")
        model_zip = os.path.join(config.model_dir, sub, "best_model.zip")
        if not os.path.exists(model_zip):
            continue

        print(f"\n▶ Evaluate Stage {stage}")
        bm = SAC.load(model_zip)
        eval_env = create_env(config.env_name)

        # 보상 & 길이 평가
        rewards, lengths = evaluate_policy(
            bm, eval_env, 
            n_eval_episodes=20, 
            deterministic=True, 
            return_episode_rewards=True
        )
        # 성공률 계산
        succ = 0
        for _ in range(20):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                a, _ = bm.predict(obs, deterministic=True)
                obs, _, term, trunc, info = eval_env.step(a)
                done = term or trunc
            if info.get('is_success', False):
                succ += 1

        comparison_results[stage] = {
            'mean_reward': np.mean(rewards),
            'std_reward' : np.std(rewards),
            'mean_length': np.mean(lengths),
            'success_rate': succ / 20,
            'rewards': rewards
        }

    # 2) Final Model 평가
    print("\n▶ Evaluate Final Model")
    eval_env = create_env(config.env_name)
    fr, fl = evaluate_policy(
        final_model, eval_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=True
    )
    fs = 0
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            a, _ = final_model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(a)
            done = term or trunc
        if info.get('is_success', False):
            fs += 1

    comparison_results['final'] = {
        'mean_reward': np.mean(fr),
        'std_reward' : np.std(fr),
        'mean_length': np.mean(fl),
        'success_rate': fs / 20,
        'rewards': fr
    }

    return comparison_results

# def fixed_compare_best_vs_final_models(final_model, training_callback):
#     """🎯 수정된 Best Model vs Final Model 성능 비교"""
#     print("\n🔍 수정된 Best Model vs Final Model 성능 비교...")
    
#     # Best model 로드
#     best_model_path = None
#     model_dir = config.model_dir
    
#     # Best model 파일 찾기
#     for root, dirs, files in os.walk(model_dir):
#         if "best_model" in root:
#             for file in files:
#                 if file.endswith('.zip'):
#                     best_model_path = os.path.join(root, file)
#                     break
#             if best_model_path:
#                 break
    
#     if not best_model_path or not os.path.exists(best_model_path):
#         print("❌ Best model을 찾을 수 없습니다. 비교를 건너뜁니다.")
#         return None
    
#     try:
#         print(f"📁 Best model 로드: {best_model_path}")
#         best_model = SAC.load(best_model_path)
        
#         # 🔧 Eval과 동일한 환경 생성
#         eval_env = create_env(config.env_name)
        
#         print("🔄 성능 평가 중...")
        
#         # Best model 평가
#         print("   Best Model 평가 중...")
#         best_rewards, best_lengths = evaluate_policy(
#             best_model, eval_env, 
#             n_eval_episodes=50, 
#             deterministic=True, 
#             return_episode_rewards=True
#         )
        
#         # Final model 평가  
#         print("   Final Model 평가 중...")
#         final_rewards, final_lengths = evaluate_policy(
#             final_model, eval_env,
#             n_eval_episodes=50,
#             deterministic=True,
#             return_episode_rewards=True
#         )
        
#         # 🎯 실제 성공률 계산 (evaluate_policy의 info로부터)
#         # 실제 성공 에피소드 카운트를 위한 추가 평가
#         best_success_count = 0
#         final_success_count = 0
        
#         # Best model 성공률 측정
#         for _ in range(50):
#             obs, _ = eval_env.reset()
#             done = False
#             while not done:
#                 action, _ = best_model.predict(obs, deterministic=True)
#                 obs, reward, terminated, truncated, info = eval_env.step(action)
#                 done = terminated or truncated
#                 if done and info.get('is_success', False):
#                     best_success_count += 1
#                     break
        
#         # Final model 성공률 측정
#         for _ in range(50):
#             obs, _ = eval_env.reset()
#             done = False
#             while not done:
#                 action, _ = final_model.predict(obs, deterministic=True)
#                 obs, reward, terminated, truncated, info = eval_env.step(action)
#                 done = terminated or truncated
#                 if done and info.get('is_success', False):
#                     final_success_count += 1
#                     break
        
#         best_success_rate = best_success_count / 50
#         final_success_rate = final_success_count / 50
        
#         # 결과 정리
#         comparison_results = {
#             'best_model': {
#                 'mean_reward': np.mean(best_rewards),
#                 'std_reward': np.std(best_rewards),
#                 'mean_length': np.mean(best_lengths),
#                 'success_rate': best_success_rate,  # ✅ 실제 성공률
#                 'rewards': best_rewards
#             },
#             'final_model': {
#                 'mean_reward': np.mean(final_rewards),
#                 'std_reward': np.std(final_rewards),
#                 'mean_length': np.mean(final_lengths),
#                 'success_rate': final_success_rate,  # ✅ 실제 성공률
#                 'rewards': final_rewards
#             }
#         }
        
#         # 결과 출력
#         print("\n" + "="*60)
#         print("📊 수정된 BEST MODEL vs FINAL MODEL 비교 결과")
#         print("="*60)
        
#         print(f"🏆 Best Model:")
#         print(f"   평균 보상: {comparison_results['best_model']['mean_reward']:.2f} ± {comparison_results['best_model']['std_reward']:.2f}")
#         print(f"   실제 성공률: {comparison_results['best_model']['success_rate']:.3f}")
#         print(f"   평균 길이: {comparison_results['best_model']['mean_length']:.1f}")
        
#         print(f"\n🎯 Final Model:")
#         print(f"   평균 보상: {comparison_results['final_model']['mean_reward']:.2f} ± {comparison_results['final_model']['std_reward']:.2f}")
#         print(f"   실제 성공률: {comparison_results['final_model']['success_rate']:.3f}")
#         print(f"   평균 길이: {comparison_results['final_model']['mean_length']:.1f}")
        
#         # 승자 판정
#         best_better_reward = comparison_results['best_model']['mean_reward'] > comparison_results['final_model']['mean_reward']
#         best_better_success = comparison_results['best_model']['success_rate'] > comparison_results['final_model']['success_rate']
        
#         print(f"\n🏅 종합 판정:")
#         if best_better_reward and best_better_success:
#             print("   🥇 Best Model이 더 우수합니다!")
#         elif not best_better_reward and not best_better_success:
#             print("   🥇 Final Model이 더 우수합니다!")
#         else:
#             print("   🤝 두 모델이 각각 장단점이 있습니다!")
        
#         # 시각화 생성
#         create_comparison_visualization(comparison_results)
        
#         # 🎯 수정된 비교 비디오 생성
#         create_fixed_comparison_videos(best_model, final_model, training_callback)
        
#         print("="*60)
        
#         return comparison_results
        
    # except Exception as e:
    #     print(f"❌ 모델 비교 중 오류 발생: {e}")
    #     return None

def create_comparison_visualization(comparison_results):
    """Best vs Final 모델 비교 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        # 결과 저장 디렉토리
        viz_dir = os.path.join(config.results_dir, "model_comparison")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 보상 분포 비교 (박스플롯)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 박스플롯
        data_to_plot = [comparison_results['best_model']['rewards'], 
                       comparison_results['final_model']['rewards']]
        ax1.boxplot(data_to_plot, labels=['Best Model', 'Final Model'])
        ax1.set_title('보상 분포 비교')
        ax1.set_ylabel('보상')
        ax1.grid(True, alpha=0.3)
        
        # 히스토그램
        ax2.hist(comparison_results['best_model']['rewards'], alpha=0.7, label='Best Model', bins=20)
        ax2.hist(comparison_results['final_model']['rewards'], alpha=0.7, label='Final Model', bins=20)
        ax2.set_title('보상 히스토그램')
        ax2.set_xlabel('보상')
        ax2.set_ylabel('빈도')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'reward_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 성능 요약 막대 그래프
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ['Best Model', 'Final Model']
        rewards = [comparison_results['best_model']['mean_reward'], 
                  comparison_results['final_model']['mean_reward']]
        success_rates = [comparison_results['best_model']['success_rate'], 
                        comparison_results['final_model']['success_rate']]
        
        ax1.bar(models, rewards, color=['gold', 'skyblue'])
        ax1.set_title('평균 보상 비교')
        ax1.set_ylabel('평균 보상')
        for i, v in enumerate(rewards):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
        
        ax2.bar(models, success_rates, color=['gold', 'skyblue'])
        ax2.set_title('실제 성공률 비교')
        ax2.set_ylabel('성공률')
        ax2.set_ylim(0, 1.0)
        for i, v in enumerate(success_rates):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 비교 그래프 저장: {viz_dir}")
        
    except Exception as e:
        print(f"⚠️ 시각화 생성 오류: {e}")

# def create_fixed_comparison_videos(best_model, final_model, training_callback):
#     """🎯 수정된 Best vs Final 모델 비교 비디오 생성"""
#     try:
#         print("🎥 수정된 Best vs Final 모델 비교 비디오 생성 중...")
        
#         # 비교 비디오 디렉토리
#         comparison_dir = os.path.join(config.video_dir, "model_comparison")
#         os.makedirs(comparison_dir, exist_ok=True)
        
#         # 🔧 수정된 비디오 시스템 사용
#         fixed_video_system = FixedStageBasedVideoSystem(
#             base_dir=comparison_dir,
#             total_timesteps=config.total_timesteps
#         )
        
#         print("   Best Model 에피소드 녹화 중...")
#         fixed_video_system.record_stage_episodes(
#             stage_name="best_model",
#             model=best_model,
#             eval_env_stats=training_callback.eval_env_stats,
#             num_episodes=3
#         )
        
#         print("   Final Model 에피소드 녹화 중...")
#         fixed_video_system.record_stage_episodes(
#             stage_name="final_model", 
#             model=final_model,
#             eval_env_stats=training_callback.eval_env_stats,
#             num_episodes=3
#         )
        
#         print(f"✅ 수정된 비교 비디오 완료: {comparison_dir}")
        
#     except Exception as e:
#         print(f"⚠️ 비교 비디오 생성 오류: {e}")

def create_fixed_comparison_videos(best_model, final_model, training_callback):
    print("🎥 시점별 Best vs Final 모델 비교 비디오 생성 중...")
    comparison_dir = os.path.join(config.video_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    video_sys = FixedStageBasedVideoSystem(
        base_dir=comparison_dir,
        total_timesteps=config.total_timesteps
    )

    # 1) best_model_* 순회
    for sub in sorted(os.listdir(config.model_dir)):
        if not sub.startswith("best_model_"):
            continue
        stage = sub.replace("best_model_", "")
        print(f"   ▶ Best Model [{stage}] 녹화")
        model_path = os.path.join(config.model_dir, sub, "best_model.zip")
        bm = SAC.load(model_path)
        video_sys.record_stage_episodes(
            stage_name=f"best_{stage}",
            model=bm,
            eval_env_stats=training_callback.eval_env_stats,
            num_episodes=config.episodes_per_stage
        )

    # 2) final 모델도 동일하게 녹화
    print("   ▶ Final Model 녹화")
    video_sys.record_stage_episodes(
        stage_name="final",
        model=final_model,
        eval_env_stats=training_callback.eval_env_stats,
        num_episodes=config.episodes_per_stage
    )

    print(f"✅ 비교 비디오 생성 완료: {comparison_dir}")


def record_final_videos(model, training_callback):
    """🎯 수정된 최종 비디오 녹화"""
    print("\n🎥 수정된 최종 성능 비디오 녹화...")
    
    # 최종 단계 녹화 (만약 아직 안 됐다면)
    final_stage = '5_100percent'
    if final_stage not in training_callback.video_system.completed_stages:
        training_callback.video_system.record_stage_episodes(
            stage_name=final_stage,
            model=model,
            eval_env_stats=training_callback.eval_env_stats,
            num_episodes=5  # 최종은 더 많이
        )
    
    # 하이라이트 비디오 생성
    training_callback.video_system.create_highlight_video()
    
    # 🎯 수정된 Best vs Final 모델 비교
    comparison_results = fixed_compare_best_vs_final_models(model, training_callback)
    
    print(f"✅ 수정된 최종 비디오 완료!")

def train_model():
    """🎯 수정된 모델 학습"""
    print(f"🎯 환경: {config.env_name}")
    print(f"🧠 알고리즘: {config.algorithm}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
    print(f"🎥 수정된 에피소드별 비디오: 각 단계별 {config.episodes_per_stage}개")
    print(f"🔧 보상 스케일링: {config.reward_scale}")
    print(f"📈 6단계 체계적 시각화 (0%, 20%, 40%, 60%, 80%, 100%)")
    print(f"✅ Eval 환경과 완전히 일치하는 비디오")
    print(f"✅ 실제 info['is_success'] 기준 사용")
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
    
    # 🎯 수정된 학습 모니터링
    training_callback = FixedTrainingCallback(config.log_dir)
    training_callback.model = model
    callbacks.append(training_callback)
    
    # 🔧 수정된 평가 콜백
    class FixedEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            self.training_callback = kwargs.pop('training_callback', None)
            super().__init__(*args, **kwargs)
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            # Eval 환경 통계를 training_callback에 전달
            if self.training_callback and hasattr(self.eval_env, 'obs_rms'):
                self.training_callback.set_eval_env_stats(self.eval_env)

            # 3) 현재 timestep 기준 스테이지 계산
            current_stage = self.training_callback.video_system.should_record_stage(self.num_timesteps)
            if current_stage:
                # 디렉토리 예: models/best_model_1_20percent/best_model.zip
                save_dir = os.path.join(config.model_dir, f"best_model_{current_stage}")
                os.makedirs(save_dir, exist_ok=True)
                # EvalCallback 내부에서 사용하는 경로를 동적으로 교체
                self.best_model_save_path = save_dir

            return result
    
    eval_callback = FixedEvalCallback(
        eval_env,
        training_callback=training_callback,
        best_model_save_path=os.path.join(config.model_dir, "best_model"),  # prefix 용으로만 남김
        log_path=os.path.join(config.log_dir, "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.eval_deterministic,
        verbose=1
    )


    callbacks.append(eval_callback)
    
    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(config.model_dir, "checkpoints"),
        name_prefix=f"{config.algorithm}_fixed_{config.experiment_name}"
    )
    callbacks.append(checkpoint_callback)
    
    # 학습 전 랜덤 비디오
    print("\n🎬 [학습 전] 수정된 무작위 행동 에피소드 녹화!")
    training_callback.record_stage_videos("0_random")
    
    # 학습 시작
    print(f"\n🚀 수정된 {config.algorithm} 학습 시작!")
    print("=" * 60)
    print("💡 Eval 환경과 완전히 일치하는 비디오!")
    print("💡 VecNormalize 통계 동기화!")
    print("💡 실제 info['is_success'] 기준 사용!")
    print("💡 성공 에피소드 우선 녹화!")
    print("💡 Best Model 기반 비교!")
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
    final_model_path = os.path.join(config.model_dir, f"{config.algorithm}_fixed_{config.experiment_name}_final")
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
    """🎯 수정된 메인 실행 함수"""
    print(f"🚀 완전히 수정된 에피소드 기반 비디오 학습 시작!")
    print(f"🎯 목표: FrankaSlideDense 태스크 마스터하기")
    print(f"🔧 주요 수정사항:")
    print(f"   ✅ Eval 환경과 완전히 일치하는 비디오 환경")
    print(f"   ✅ VecNormalize 통계 동기화")
    print(f"   ✅ 실제 info['is_success'] 기준 사용")
    print(f"   ✅ 성공 에피소드 우선 녹화")
    print(f"   ✅ Best Model 기반 비교 시스템")
    print(f"   ✅ 각 단계별 {config.episodes_per_stage}개 에피소드")
    print(f"   ✅ 보상 스케일링: {config.reward_scale} (안정적 학습)")
    print("=" * 60)
    
    # 학습 실행
    model, training_callback = train_model()
    
    # 최종 비디오 녹화
    record_final_videos(model, training_callback)
    
    print("\n" + "=" * 60)
    print("🎉 완전히 수정된 학습 시스템 완료!")
    print(f"📊 최종 성공률: {training_callback.recent_success_rate:.3f}")
    print(f"🏆 최고 보상: {training_callback.best_reward:.2f}")
    print(f"🎥 수정된 에피소드별 비디오 시스템 완료!")
    print(f"📁 비디오 저장 위치: {config.video_dir}")
    print("📋 생성된 결과:")
    print("   ✅ 0_random/: 학습 전 무작위 행동 (수정됨)")
    print("   ✅ 1_20percent/: 20% 학습 진행 시점 (Eval 일치)")
    print("   ✅ 2_40percent/: 40% 학습 진행 시점 (Eval 일치)")
    print("   ✅ 3_60percent/: 60% 학습 진행 시점 (Eval 일치)") 
    print("   ✅ 4_80percent/: 80% 학습 진행 시점 (Eval 일치)")
    print("   ✅ 5_100percent/: 100% 학습 완료 시점 (Eval 일치)")
    print("   ✅ highlights/: 최고 보상 에피소드들 (실제 성공 기준)")
    print("   ✅ model_comparison/: Best vs Final 모델 비교 (수정됨)")
    print(f"📊 성능 비교 그래프: {config.results_dir}/model_comparison/")
    print("🎯 이제 비디오가 실제 학습 성능을 정확히 반영합니다!")
    print("=" * 60)
    
    return model, training_callback

if __name__ == "__main__":
    model, training_callback = main()
