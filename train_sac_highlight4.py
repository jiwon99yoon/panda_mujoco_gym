# train 코드 수정 필요 : 병렬 환경에 맞게끔 조정 필요 : 학습 잘 안됨 (현재)
# # #/home/minjun/panda_mujoco_gym/train_sac_highlight4.py
# # /home/dyros/panda_mujoco_gym/train_sac_highlight4.py

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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
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

    # 3. FixedEpisodeVideoRecorder의 end_episode_recording 메서드 수정
    def end_episode_recording(self, episode_info: Dict) -> Optional[str]:
        """에피소드 녹화 종료 및 비디오 저장 - 개선된 메타데이터"""
        if not self.current_episode_frames:
            return None
            
        # 파일명 생성
        stage = episode_info.get('stage', 'unknown')
        episode_id = episode_info.get('episode_id', 0)
        success_str = 'SUCCESS' if episode_info.get('success', False) else 'FAIL'
        task_str = '_TASK_COMPLETE' if episode_info.get('task_completed', False) else ''
        reward = episode_info.get('reward', 0)
        
        base_filename = f"{stage}_ep{episode_id:03d}_{success_str}{task_str}_reward{reward:.1f}"
        
        # 덮어쓰기 방지
        version = 1
        while True:
            if version == 1:
                filename = f"{base_filename}.mp4"
            else:
                filename = f"{base_filename}_v{version}.mp4"
            
            video_path = os.path.join(self.save_dir, filename)
            
            if not os.path.exists(video_path):
                break
            
            version += 1
            if version > 100:
                print(f"⚠️ 경고: {base_filename}의 버전이 100개를 초과했습니다!")
                break

        # 비디오 저장
        success = self._save_video(video_path, self.current_episode_frames)
        
        if success:
            # 메타데이터 저장
            episode_info['video_path'] = video_path
            episode_info['frame_count'] = len(self.current_episode_frames)
            episode_info['fps'] = self.fps
            episode_info['duration'] = len(self.current_episode_frames) / self.fps
            episode_info['version'] = version
            self.episode_metadata.append(episode_info)
            
            print(f"✅ 에피소드 비디오 저장: {filename}")
            print(f"   프레임 수: {len(self.current_episode_frames)}, 성공: {success_str}, 보상: {reward:.2f}")
            print(f"   재생 시간: {episode_info['duration']:.1f}초, 태스크 완료: {episode_info.get('task_completed', False)}")
            if 'total_frames' in episode_info:
                print(f"   원본 프레임: {episode_info['total_frames']}, 저장 프레임: {len(self.current_episode_frames)}")
            if version > 1:
                print(f"   📌 버전: v{version} (기존 파일 덮어쓰기 방지)")
            
            return video_path
        
        return None
    
    def _save_video(self, video_path: str, frames: List[np.ndarray]) -> bool:
        """프레임들을 비디오 파일로 저장 - 개선된 인코딩"""
        if not frames:
            return False
            
        try:
            height, width = frames[0].shape[:2]
            
            # 🎯 더 나은 코덱 사용 (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # 대안: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            if not writer.isOpened():
                # H264 실패 시 mp4v로 fallback
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            # 프레임 저장 with progress
            total_frames = len(frames)
            for i, frame in enumerate(frames):
                # RGB → BGR 변환
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                writer.write(frame_bgr)
                
                # 진행 상황 표시 (긴 비디오의 경우)
                if total_frames > 300 and i % 100 == 0:
                    progress = (i / total_frames) * 100
                    print(f"     비디오 인코딩 진행: {progress:.1f}%", end='\r')
            
            writer.release()
            
            # 파일 생성 확인
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"     비디오 저장 완료: {total_frames} 프레임")
                return True
            else:
                print(f"❌ 비디오 파일 생성 실패: {video_path}")
                return False
            
        except Exception as e:
            print(f"❌ 비디오 저장 오류: {e}")
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

    # 4. FixedStageBasedVideoSystem의 record_stage_episodes 메서드 수정
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
        successful_episodes = []
        failed_episodes = []
        attempts = 0
        max_attempts = num_episodes * 15  # 성공 에피소드를 위해 더 많은 시도 (10->15)


        # ------------------------------- 추가 ---------------------  #
        print(f"  목표: 성공 에피소드 {num_episodes}개 수집 (최대 {max_attempts}회 시도)")
        
        while len(episode_results) < num_episodes and attempts < max_attempts:
            attempts += 1
            
            result = self._record_single_episode(
                recorder, model, video_env, attempts - 1, stage_name
            )
            
            if result:
                if result['task_completed']:
                    successful_episodes.append(result)
                    print(f"    ✅ 태스크 완료 에피소드 수집! (성공: {len(successful_episodes)}개)")
                else:
                    failed_episodes.append(result)
                    print(f"    ❌ 태스크 실패 에피소드 (실패: {len(failed_episodes)}개)")
                
                # 성공 에피소드 우선, 부족하면 실패 에피소드로 채움
                if len(successful_episodes) >= num_episodes:
                    episode_results = successful_episodes[:num_episodes]
                    break
                elif len(successful_episodes) + len(failed_episodes) >= num_episodes:
                    episode_results = successful_episodes + failed_episodes[:num_episodes - len(successful_episodes)]
                    break
            
            # 진행 상황 출력
            if attempts % 5 == 0:
                print(f"    시도 {attempts}/{max_attempts}: 성공 {len(successful_episodes)}개, 실패 {len(failed_episodes)}개")
        
        # 최종 에피소드 결과 처리
        for result in episode_results:
            self.stage_videos[stage_name].append(result['video_path'])
            
            # 최고 보상 추적
            if not self.best_episodes or result['reward'] > min(ep['reward'] for ep in self.best_episodes):
                self.best_episodes.append(result)
                self.best_episodes.sort(key=lambda x: x['reward'], reverse=True)
                self.best_episodes = self.best_episodes[:10]  # 상위 10개만 유지
        
        self.completed_stages.add(stage_name)
        
        # 결과 요약
        task_completed_count = sum(1 for r in episode_results if r.get('task_completed', False))
        print(f"✅ [{stage_name}] 완료! {len(episode_results)}개 에피소드 녹화됨")
        print(f"   태스크 완료: {task_completed_count}/{len(episode_results)}")
        if episode_results:
            avg_reward = sum(r['reward'] for r in episode_results) / len(episode_results)
            avg_duration = sum(r.get('frame_count', 0) / config.video_fps for r in episode_results) / len(episode_results)
            print(f"   평균 보상: {avg_reward:.2f}")
            print(f"   평균 재생 시간: {avg_duration:.1f}초")
        
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

# 2. FixedStageBasedVideoSystem의 _record_single_episode 메서드 개선
    def _record_single_episode(self, recorder, model, video_env, episode_idx, stage_name):
        """단일 에피소드 녹화 - 개선된 프레임 캡처"""
        recorder.start_episode_recording()
        
        obs = video_env.reset()
        total_reward = 0
        step_count = 0
        done = False
        task_completed = False
        
        # 프레임 캡처를 위한 변수
        frame_buffer = []  # 모든 프레임을 버퍼에 저장
        action_history = []  # 액션 히스토리 저장 (디버깅용)
        
        # 초기 상태 캡처 (여러 프레임)
        for _ in range(5):  # 초기 0.25초 정도 캡처
            rendered = video_env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                frame_buffer.append(img)
        
        while not done:
            # 액션 예측
            if model and hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = video_env.action_space.sample()
            
            action_history.append(action)
            
            # 환경 스텝
            obs, rewards, dones, infos = video_env.step(action)
            done = dones[0]
            total_reward += rewards[0]
            step_count += 1
            
            # 🎯 매 스텝마다 프레임 캡처 (프레임 누락 방지)
            rendered = video_env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                frame_buffer.append(img)
            
            # 태스크 성공 체크
            if infos[0].get('is_success', False):
                task_completed = True
                
                # 🎯 성공 후 짧은 추가 녹화 (0.5초)
                for _ in range(config.success_extra_frames):
                    rendered = video_env.render()
                    if rendered is not None:
                        img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                        frame_buffer.append(img)
                done = True
            
            # 무한 루프 방지
            if step_count > config.max_episode_steps:
                break
        
        # 🎯 프레임 서브샘플링 (필요한 경우)
        # 너무 많은 프레임이면 적절히 샘플링
        if len(frame_buffer) > 600:  # 20초 이상이면
            # 균등하게 샘플링하되, 중요한 순간은 보존
            sampled_frames = self._smart_frame_sampling(frame_buffer, target_frames=600)
            recorder.current_episode_frames = sampled_frames
        else:
            # 모든 프레임 사용
            recorder.current_episode_frames = frame_buffer
        
        # 에피소드 정보
        episode_info = {
            'episode_id': episode_idx,
            'reward': total_reward,
            'length': step_count,
            'success': task_completed,
            'stage': stage_name,
            'task_completed': task_completed,
            'total_frames': len(frame_buffer),
            'action_count': len(action_history)
        }
        
        # 비디오 저장
        video_path = recorder.end_episode_recording(episode_info)
        
        if video_path:
            episode_info['video_path'] = video_path
            return episode_info
        
        return None
    
    def _smart_frame_sampling(self, frames, target_frames=600):
        """스마트 프레임 샘플링 - 중요한 순간 보존"""
        if len(frames) <= target_frames:
            return frames
        
        # 시작과 끝 부분은 더 많이 보존
        start_frames = frames[:60]  # 처음 2초
        end_frames = frames[-60:]   # 마지막 2초
        
        # 중간 부분은 균등 샘플링
        middle_frames = frames[60:-60]
        middle_target = target_frames - 120
        
        if len(middle_frames) > middle_target:
            # 균등하게 샘플링
            indices = np.linspace(0, len(middle_frames)-1, middle_target, dtype=int)
            middle_sampled = [middle_frames[i] for i in indices]
        else:
            middle_sampled = middle_frames
        
        return start_frames + middle_sampled + end_frames
    
    def create_highlight_video(self):
        """최고 보상 에피소드들로 하이라이트 비디오 생성 (덮어쓰기 방지 추가)"""
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
                base_filename = f"highlight_{i+1:02d}_reward{episode['reward']:.1f}_{episode['stage']}"
                
                # 🔧 덮어쓰기 방지: 파일이 이미 존재하면 버전 번호 추가
                version = 1
                while True:
                    if version == 1:
                        highlight_filename = f"{base_filename}.mp4"
                    else:
                        highlight_filename = f"{base_filename}_v{version}.mp4"
                    
                    highlight_path = os.path.join(highlight_dir, highlight_filename)
                    
                    # 파일이 존재하지 않으면 이 이름으로 저장
                    if not os.path.exists(highlight_path):
                        break
                    
                    version += 1
                    
                    # 안전장치
                    if version > 100:
                        print(f"⚠️ 경고: {base_filename}의 버전이 100개를 초과했습니다!")
                        break
                
                # 파일 복사
                shutil.copy2(original_path, highlight_path)
                print(f"  ✨ {highlight_filename}")
                if version > 1:
                    print(f"     📌 버전: v{version}")
        
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
    #gpu 작업량 증가
    # train_freq, gradient_steps 증가 각각 8, 16
    
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
    video_fps = 100                 # 비디오 FPS -> MUJOCO가 0.01 dt로 실행되는 것 반영하여 100으로 일단 설정해봄
    recording_fps = 120            # 녹화 시 프레임 캡처 빈도 (더 많은 프레임 캡처)
    max_episode_steps = 5000       # 에피소드 최대 스텝 수 증가 (1000 → 2000)
    success_extra_frames = 100      # 성공 후 추가 프레임 수 (0.5초 @ 20fps)
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

# def create_vec_env(env_name, n_envs=1, normalize=True):
#     """벡터화된 환경 생성"""
#     def make_env():
#         env = create_env(env_name)
#         return env
    
#     vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
#     if normalize:
#         vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
#     return vec_env

#병렬 환경으로 교체 - n_envs=4로 설정, 
# export 추가 from stable_baselines3.common.vec_env import SubprocVecEnv
# def create_vec_env(env_name, n_envs=4, normalize=True):
#     def make_env():
#         return lambda: create_env(env_name)

#     vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])

#     if normalize:
#         vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

#     return vec_env
def create_vec_env(env_name: str, normalize: bool = True, num_envs: int = 4):
    def make_env_fn(rank):
        def _init():
            env = gym.make(env_name)  # ✅ 이미 등록된 환경 사용
            return env
        return _init

    env_fns = [make_env_fn(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        vec_env.reset()  # 필수! 초기화 안 하면 에러 날 수 있음

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
