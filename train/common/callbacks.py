"""
학습 콜백 클래스
"""

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """학습 진행 상황 추적 콜백"""
    
    def __init__(self, config, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.config = config
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.csv_file = os.path.join(config.log_dir, "training_log.csv")
        
        # 성능 추적
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.recent_success_rate = 0.0
        
        # 단계 저장을 위한 변수
        self.saved_stages = set()
        self.stage_timesteps = config.get_stage_timesteps()
        
        # CSV 파일 초기화
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestep', 'Episode', 'Reward', 'Length', 
                'Success', 'Success_Rate', 'Best_Reward', 'Stage'
            ])
    
    def _on_step(self) -> bool:
        # 단계별 모델 저장 체크
        self._check_stage_save()
        
        # 에피소드 완료 시 처리
        if len(self.locals.get('dones', [])) > 0 and self.locals.get('dones', [False])[0]:
            self._handle_episode_end()
        
        return True
    
    def _check_stage_save(self):
        """단계별 모델 저장"""
        current_timestep = self.num_timesteps
        
        for stage_name, stage_timestep in self.stage_timesteps.items():
            if (stage_name not in self.saved_stages and 
                current_timestep >= stage_timestep):
                
                # 모델을 models 폴더에 직접 저장
                model_path = os.path.join(
                    self.config.model_dir, 
                    f"stage_{stage_name}.zip"
                )
                self.model.save(model_path)
                
                # 통계는 logs 폴더에 저장
                stats_path = os.path.join(
                    self.config.log_dir, 
                    f"stage_{stage_name}_stats.npz"
                )
                np.savez(
                    stats_path,
                    timestep=current_timestep,
                    episode_count=self.episode_count,
                    success_rate=self.recent_success_rate,
                    best_reward=self.best_reward,
                    recent_rewards=self.recent_rewards[-100:]
                )
                
                self.saved_stages.add(stage_name)
                print(f"💾 단계 모델 저장: {stage_name} (Step {current_timestep})")
    
    def _handle_episode_end(self):
        """에피소드 종료 처리"""
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
            
            # 현재 단계 확인
            current_stage = self._get_current_stage()
            
            # 로깅
            if self.episode_count % 10 == 0:
                self._print_progress()
            
            # CSV 로그 저장
            self._save_to_csv(episode_reward, episode_length, is_success, current_stage)
    
    def _get_current_stage(self) -> str:
        """현재 학습 단계 반환"""
        current_timestep = self.num_timesteps
        current_stage = '0_random'
        
        for stage_name, stage_timestep in sorted(self.stage_timesteps.items(), key=lambda x: x[1]):
            if current_timestep >= stage_timestep:
                current_stage = stage_name
            else:
                break
                
        return current_stage
    
    def _print_progress(self):
        """진행 상황 출력"""
        avg_reward = np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
        avg_length = np.mean(self.episode_lengths[-50:]) if len(self.episode_lengths) >= 50 else np.mean(self.episode_lengths)
        
        print(f"📊 Episode {self.episode_count:4d} | "
              f"Step {self.num_timesteps:7d} | "
              f"Reward: {self.recent_rewards[-1]:7.2f} | "
              f"Avg: {avg_reward:7.2f} | "
              f"Success: {self.recent_success_rate:.3f}")
    
    def _save_to_csv(self, episode_reward, episode_length, is_success, current_stage):
        """CSV 파일에 로그 저장"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                self.episode_count,
                episode_reward,
                episode_length,
                is_success,
                self.recent_success_rate,
                self.best_reward,
                current_stage
            ])
