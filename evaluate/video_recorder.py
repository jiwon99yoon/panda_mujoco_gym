#/home/dyros/panda_mujoco_gym/evaluate/video_recorder.py
"""
비디오 녹화 클래스들
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
import json

class EpisodeVideoRecorder:
    """에피소드별 개별 비디오 녹화 클래스"""
    
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
        """에피소드 녹화 종료 및 비디오 저장"""
        if not self.current_episode_frames:
            return None
            
        # 파일명 생성
        stage = episode_info.get('stage', 'unknown')
        episode_id = episode_info.get('episode_id', 0)
        success_str = 'SUCCESS' if episode_info.get('success', False) else 'FAIL'
        reward = episode_info.get('reward', 0)
        
        filename = f"ep{episode_id:03d}_{success_str}_reward{reward:.1f}.mp4"
        video_path = os.path.join(self.save_dir, filename)
        
        # 덮어쓰기 방지
        version = 1
        while os.path.exists(video_path) and version < 100:
            filename = f"ep{episode_id:03d}_{success_str}_reward{reward:.1f}_v{version}.mp4"
            video_path = os.path.join(self.save_dir, filename)
            version += 1
        
        # 비디오 저장
        success = self._save_video(video_path, self.current_episode_frames)
        
        if success:
            # 메타데이터 저장
            episode_info['video_path'] = video_path
            episode_info['frame_count'] = len(self.current_episode_frames)
            episode_info['fps'] = self.fps
            episode_info['duration'] = len(self.current_episode_frames) / self.fps
            self.episode_metadata.append(episode_info)
            
            print(f"   ✅ 비디오 저장: {filename}")
            print(f"      프레임: {len(self.current_episode_frames)}, 성공: {success_str}, 보상: {reward:.2f}")
            
            return video_path
        
        return None
    
    def _save_video(self, video_path: str, frames: List[np.ndarray]) -> bool:
        """프레임들을 비디오 파일로 저장"""
        if not frames:
            return False
            
        try:
            height, width = frames[0].shape[:2]
            
            # 코덱 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            if not writer.isOpened():
                print(f"❌ 비디오 writer 생성 실패")
                return False
            
            # 프레임 저장
            for frame in frames:
                # RGB → BGR 변환
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                writer.write(frame_bgr)
            
            writer.release()
            
            # 파일 생성 확인
            return os.path.exists(video_path) and os.path.getsize(video_path) > 0
            
        except Exception as e:
            print(f"❌ 비디오 저장 오류: {e}")
            return False
    
    def save_metadata(self):
        """에피소드 메타데이터를 JSON으로 저장"""
        if self.episode_metadata:
            metadata_path = os.path.join(self.save_dir, 'episode_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.episode_metadata, f, indent=4)
            print(f"   📋 메타데이터 저장: {metadata_path}")


class StageVideoRecorder:
    """학습 단계별 비디오 녹화 시스템"""
    
    def __init__(self, base_dir: str, fps: int = 30):
        self.base_dir = base_dir
        self.fps = fps
        self.stage_results = {}
        self.best_episodes = []
        os.makedirs(base_dir, exist_ok=True)
    
    def record_stage_episodes(self, stage_name: str, model, env, num_episodes: int = 3):
        """특정 단계의 에피소드들 녹화"""
        print(f"\n🎬 [{stage_name}] 비디오 녹화 시작 ({num_episodes}개 에피소드)")
        
        # 단계별 디렉토리 생성
        stage_dir = os.path.join(self.base_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        
        recorder = EpisodeVideoRecorder(stage_dir, fps=self.fps)
        
        episode_results = []
        successful_episodes = []
        failed_episodes = []
        
        attempts = 0
        max_attempts = num_episodes * 5  # 충분한 시도 횟수
        
        while len(episode_results) < num_episodes and attempts < max_attempts:
            attempts += 1
            
            result = self._record_single_episode(
                recorder, model, env, len(episode_results), stage_name
            )
            
            if result:
                if result['success']:
                    successful_episodes.append(result)
                else:
                    failed_episodes.append(result)
                
                # 성공 에피소드 우선 수집
                if len(successful_episodes) >= num_episodes:
                    episode_results = successful_episodes[:num_episodes]
                    break
                elif attempts >= max_attempts - 1:  # 마지막 시도
                    # 성공 에피소드 + 실패 에피소드로 채우기
                    episode_results = successful_episodes + failed_episodes
                    episode_results = episode_results[:num_episodes]
                    break
        
        # 메타데이터 저장
        recorder.save_metadata()
        
        # 결과 저장
        self.stage_results[stage_name] = episode_results
        
        # 최고 보상 에피소드 추적
        for result in episode_results:
            if not self.best_episodes or result['reward'] > min(ep['reward'] for ep in self.best_episodes):
                self.best_episodes.append(result)
                self.best_episodes.sort(key=lambda x: x['reward'], reverse=True)
                self.best_episodes = self.best_episodes[:10]  # 상위 10개만 유지
        
        # 결과 요약
        success_count = sum(1 for r in episode_results if r.get('success', False))
        avg_reward = np.mean([r['reward'] for r in episode_results]) if episode_results else 0
        
        print(f"✅ [{stage_name}] 완료!")
        print(f"   녹화된 에피소드: {len(episode_results)}개")
        print(f"   성공률: {success_count}/{len(episode_results)}")
        print(f"   평균 보상: {avg_reward:.2f}")
        
        return episode_results
    
    def _record_single_episode(self, recorder, model, env, episode_idx, stage_name):
        """단일 에피소드 녹화"""
        recorder.start_episode_recording()
        
        obs = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        success = False
        
        # 초기 프레임 캡처
        for _ in range(3):  # 시작 시 몇 프레임 추가
            rendered = env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                recorder.add_frame(img)
        
        while not done:
            # 액션 예측
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Random policy for stage_0_random: batch 차원(1, action_dim)으로 감싸 줍니다.
                raw_action = env.action_space.sample()      # shape: (action_dim,)
                action = np.array([raw_action])            # shape: (1, action_dim)
            
            # 환경 스텝
            obs, rewards, dones, infos = env.step(action)
            done = dones[0]
            total_reward += rewards[0]
            step_count += 1
            
            # 프레임 추가
            rendered = env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                recorder.add_frame(img)
            
            # 성공 체크
            if infos[0].get('is_success', False):
                success = True
                # 성공 후 추가 프레임
                for _ in range(15):  # 0.5초 추가 (30fps 기준)
                    rendered = env.render()
                    if rendered is not None:
                        img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                        recorder.add_frame(img)
                break
            
            # 무한 루프 방지
            if step_count > 1000:
                break
        
        # 에피소드 정보
        episode_info = {
            'episode_id': episode_idx,
            'reward': total_reward,
            'length': step_count,
            'success': success,
            'stage': stage_name,
        }
        
        # 비디오 저장
        video_path = recorder.end_episode_recording(episode_info)
        
        if video_path:
            episode_info['video_path'] = video_path
            return episode_info
        
        return None
    
    def create_highlight_reel(self):
        """최고 보상 에피소드들로 하이라이트 비디오 생성"""
        if not self.best_episodes:
            print("❌ 하이라이트 생성 실패: 에피소드 없음")
            return
        
        print(f"\n🌟 하이라이트 폴더 생성 중... (상위 {len(self.best_episodes)}개 에피소드)")
        
        highlight_dir = os.path.join(self.base_dir, "highlights")
        os.makedirs(highlight_dir, exist_ok=True)
        
        # 각 최고 에피소드를 하이라이트 폴더에 복사
        import shutil
        for i, episode in enumerate(self.best_episodes):
            if 'video_path' in episode and os.path.exists(episode['video_path']):
                original_path = episode['video_path']
                highlight_filename = f"best_{i+1:02d}_reward{episode['reward']:.1f}_{episode['stage']}.mp4"
                highlight_path = os.path.join(highlight_dir, highlight_filename)
                
                shutil.copy2(original_path, highlight_path)
                print(f"   ✨ {highlight_filename}")
        
        # 하이라이트 메타데이터 저장
        highlight_metadata = {
            'best_episodes': self.best_episodes,
            'total_stages_evaluated': len(self.stage_results),
            'stages': list(self.stage_results.keys())
        }
        
        metadata_path = os.path.join(highlight_dir, 'highlight_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(highlight_metadata, f, indent=4)
        
        print(f"✅ 하이라이트 완료! {highlight_dir}")
    
    def save_evaluation_summary(self):
        """전체 평가 요약 저장"""
        summary = {
            'stages_evaluated': list(self.stage_results.keys()),
            'total_episodes': sum(len(episodes) for episodes in self.stage_results.values()),
            'stage_summaries': {}
        }
        
        for stage, episodes in self.stage_results.items():
            if episodes:
                rewards = [ep['reward'] for ep in episodes]
                successes = [ep['success'] for ep in episodes]
                
                summary['stage_summaries'][stage] = {
                    'num_episodes': len(episodes),
                    'mean_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'success_rate': sum(successes) / len(successes),
                    'best_reward': float(max(rewards)),
                    'worst_reward': float(min(rewards))
                }
        
        summary_path = os.path.join(self.base_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n📊 평가 요약 저장: {summary_path}")
        
        return summary
