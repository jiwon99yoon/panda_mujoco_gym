#/home/dyros/panda_mujoco_gym/evaluate/evaluate_with_video.py
#!/usr/bin/env python3
"""
모델 평가 및 비디오 생성 통합 스크립트
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional
from stable_baselines3.common.evaluation import evaluate_policy

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 사용자 정의 환경 등록
import panda_mujoco_gym

# Utils 임포트
from utils import (
    load_model, load_json, save_json, get_experiment_info,
    find_latest_experiment, create_vec_env
)

# Evaluate 모듈 임포트
from video_recorder import StageVideoRecorder


def evaluate_model_performance(model, env, num_episodes: int = 50) -> Dict:
    """모델 성능 평가"""
    print(f"\n📊 모델 성능 평가 중... ({num_episodes}개 에피소드)")
    
    # evaluate_policy로 기본 평가
    rewards, lengths = evaluate_policy(
        model, env,
        n_eval_episodes=num_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # 성공률 계산을 위한 추가 평가
    success_count = 0
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            if done and infos[0].get('is_success', False):
                success_count += 1
                break
    
    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes,
        'all_rewards': [float(r) for r in rewards],
        'all_lengths': [int(l) for l in lengths]
    }
    
    print(f"   평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   성공률: {results['success_rate']:.3f}")
    print(f"   평균 에피소드 길이: {results['mean_length']:.1f}")
    
    return results


def evaluate_experiment(exp_dir: str, 
                       stages: List[str] = None,
                       num_eval_episodes: int = 50,
                       num_video_episodes: int = 3,
                       record_video: bool = True,
                       create_highlights: bool = True) -> Dict:
    """
    실험 결과 평가 및 비디오 생성
    
    Args:
        exp_dir: 실험 디렉토리 경로
        stages: 평가할 단계 리스트 (None이면 모든 단계)
        num_eval_episodes: 성능 평가용 에피소드 수
        num_video_episodes: 각 단계별 비디오 에피소드 수
        record_video: 비디오 녹화 여부
        create_highlights: 하이라이트 생성 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    print("=" * 60)
    print(f"🔍 실험 평가 시작")
    print(f"📁 실험 디렉토리: {exp_dir}")
    print("=" * 60)
    
    # 실험 정보 로드
    exp_info = get_experiment_info(exp_dir)
    if not exp_info:
        raise ValueError(f"실험 정보를 찾을 수 없습니다: {exp_dir}")
    
    env_name = exp_info.get('env_name', 'FrankaSlideDense-v0')
    algorithm = exp_info.get('algorithm', 'SAC')
    reward_scale = exp_info.get('config', {}).get('reward_scale', 0.1)
    
    print(f"\n📋 실험 정보:")
    print(f"   환경: {env_name}")
    print(f"   알고리즘: {algorithm}")
    print(f"   보상 스케일: {reward_scale}")
    print(f"   총 학습 스텝: {exp_info.get('total_timesteps', 'Unknown')}")
    
    # 평가할 모델 찾기
    models_dir = os.path.join(exp_dir, 'models')
    available_models = exp_info.get('available_models', [])
    
    if not available_models:
        raise ValueError(f"모델을 찾을 수 없습니다: {models_dir}")
    
    # 평가할 stage 모델 선택
    if stages is None or 'all' in stages:
        stage_models = [m for m in available_models if m.startswith('stage_')]
        stage_models.sort()
    else:
        stage_models = []
        for stage in stages:
            model_name = f'stage_{stage}'
            if model_name in available_models:
                stage_models.append(model_name)
    
    # final 모델이 있으면 추가
    if 'final_model' in available_models and (stages is None or 'final' in stages or 'all' in stages):
        stage_models.append('final_model')
    
    print(f"\n📦 평가할 모델 ({len(stage_models)}개):")
    for model in stage_models:
        print(f"   - {model}")
    
    # 환경 생성
    vec_normalize_path = os.path.join(models_dir, 'vec_normalize.pkl')
    eval_env = create_vec_env(
        env_name,
        n_envs=1,
        normalize=True,
        reward_scale=reward_scale,
        vec_normalize_path=vec_normalize_path if os.path.exists(vec_normalize_path) else None,
        training=False,
        render_mode=None
    )
    
    # 평가 디렉토리 생성
    evaluation_dir = os.path.join(exp_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # 비디오 녹화 준비
    if record_video:
        videos_dir = os.path.join(evaluation_dir, 'videos')
        video_recorder = StageVideoRecorder(videos_dir)
    
    # 평가 결과 저장
    all_results = {
        'experiment_info': exp_info,
        'evaluation_config': {
            'num_eval_episodes': num_eval_episodes,
            'num_video_episodes': num_video_episodes,
            'stages_evaluated': stage_models
        },
        'model_performances': {},
        'video_recordings': {}
    }
    
    # 각 모델 평가
    for model_name in stage_models:
        print(f"\n{'='*50}")
        print(f"🎯 평가 중: {model_name}")
        print(f"{'='*50}")
        
        model_path = os.path.join(models_dir, f"{model_name}.zip")
        
        # 모델 로드
        if model_name == 'stage_0_random':
            # Random policy는 모델 없이 진행
            model = None
            print("   🎲 Random Policy 사용")
        else:
            model = load_model(model_path, algorithm=algorithm, env=eval_env)
            print(f"   ✅ 모델 로드 완료")
        
        # 성능 평가 (random policy 제외)
        if model is not None:
            performance = evaluate_model_performance(model, eval_env, num_eval_episodes)
            all_results['model_performances'][model_name] = performance
        else:
            print("   ⏭️  Random Policy는 성능 평가 생략")
            all_results['model_performances'][model_name] = {
                'note': 'Random policy - performance evaluation skipped'
            }
        
        # 비디오 녹화
        if record_video:
            # 비디오용 환경 생성 (렌더링 활성화)
            video_env = create_vec_env(
                env_name,
                n_envs=1,
                normalize=True,
                reward_scale=reward_scale,
                vec_normalize_path=vec_normalize_path if os.path.exists(vec_normalize_path) else None,
                training=False,
                render_mode='rgb_array'

            )
            # 렌더 모드 설정
            #video_env.env_method('set_render_mode', 'rgb_array')
            
            # stage 이름 추출
            if model_name.startswith('stage_'):
                stage_name = model_name.replace('stage_', '')
            else:
                stage_name = model_name
            
            # 에피소드 녹화
            video_results = video_recorder.record_stage_episodes(
                stage_name, model, video_env, num_video_episodes
            )
            
            all_results['video_recordings'][model_name] = {
                'num_videos': len(video_results),
                'episodes': video_results
            }
    
    # 하이라이트 생성
    if record_video and create_highlights:
        video_recorder.create_highlight_reel()
    
    # 평가 요약 저장
    if record_video:
        video_summary = video_recorder.save_evaluation_summary()
        all_results['video_summary'] = video_summary
    
    # 전체 결과 저장
    results_path = os.path.join(evaluation_dir, 'evaluation_results.json')
    save_json(all_results, results_path)
    
    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print(f"📊 결과 저장: {results_path}")
    if record_video:
        print(f"🎥 비디오 저장: {os.path.join(evaluation_dir, 'videos')}")
    print("=" * 60)
    
    return all_results


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="모델 평가 및 비디오 생성")
    
    # 실험 선택
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-dir", type=str, help="실험 디렉토리 경로")
    group.add_argument("--latest", action="store_true", help="가장 최근 실험 평가")
    
    # 평가 옵션
    parser.add_argument("--stages", type=str, nargs='+', default=None,
                       help="평가할 단계 (예: 0_random 1_20percent final) 또는 'all'")
    parser.add_argument("--num-eval", type=int, default=50,
                       help="성능 평가 에피소드 수 (기본: 50)")
    parser.add_argument("--num-video", type=int, default=3,
                       help="각 단계별 비디오 에피소드 수 (기본: 3)")
    parser.add_argument("--no-video", action="store_true",
                       help="비디오 녹화 비활성화")
    parser.add_argument("--no-highlights", action="store_true",
                       help="하이라이트 생성 비활성화")
    
    args = parser.parse_args()
    
    # 실험 디렉토리 결정
    if args.latest:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print("❌ 최근 실험을 찾을 수 없습니다.")
            return
        print(f"📁 최신 실험 발견: {exp_dir}")
    else:
        exp_dir = args.exp_dir
        if not os.path.exists(exp_dir):
            print(f"❌ 실험 디렉토리를 찾을 수 없습니다: {exp_dir}")
            return
    
    # 평가 실행
    try:
        results = evaluate_experiment(
            exp_dir=exp_dir,
            stages=args.stages,
            num_eval_episodes=args.num_eval,
            num_video_episodes=args.num_video,
            record_video=not args.no_video,
            create_highlights=not args.no_highlights
        )
        
        # 간단한 결과 요약 출력
        print("\n📈 평가 결과 요약:")
        for model_name, perf in results['model_performances'].items():
            if 'mean_reward' in perf:
                print(f"   {model_name}: 보상 {perf['mean_reward']:.2f} ± {perf['std_reward']:.2f}, "
                      f"성공률 {perf['success_rate']:.3f}")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
