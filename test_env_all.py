#/home/minjun/panda_mujoco_gym/test_env_all.py
import sys
import time
import gymnasium as gym
import panda_mujoco_gym

def test_single_env(env_id, steps=500, sleep_time=0.1):
    """단일 환경을 시각화하면서 테스트"""
    print(f"\n🎮 {env_id} 테스트 시작!")
    print("=" * 60)
    
    # 환경별 특징 설명
    descriptions = {
        "FrankaPickAndPlaceSparse-v0": "🤖 Pick & Place (희소 보상): 녹색 블록을 집어서 빨간 목표로 이동",
        "FrankaPickAndPlaceDense-v0": "🤖 Pick & Place (밀집 보상): 거리 기반 연속 보상",
        "FrankaPushSparse-v0": "👋 Push (희소 보상): 그리퍼 차단, 밀어서 목표로 이동",
        "FrankaPushDense-v0": "👋 Push (밀집 보상): 그리퍼 차단, 거리 기반 보상",
        "FrankaSlideSparse-v0": "🧊 Slide (희소 보상): 미끄러운 표면에서 퍽 치기",
        "FrankaSlideDense-v0": "🧊 Slide (밀집 보상): 미끄러운 표면, 거리 기반 보상"
    }
    
    if env_id in descriptions:
        print(f"📝 {descriptions[env_id]}")
    print()
    
    env = gym.make(env_id, render_mode="human")
    observation, info = env.reset()
    
    episode_count = 0
    success_count = 0
    total_reward = 0
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # 성공 여부 체크
        if 'is_success' in info and info['is_success']:
            success_count += 1
        
        # 주기적 상태 출력
        if i % 100 == 0:
            print(f"  Step {i:3d}: reward={reward:6.2f}, total_reward={total_reward:8.2f}")
        
        if terminated or truncated:
            episode_count += 1
            observation, info = env.reset()
            print(f"  🔄 Episode {episode_count} 완료! (Step {i})")
            total_reward = 0  # 에피소드별 리셋
        
        time.sleep(sleep_time)
    
    env.close()
    
    print(f"\n✅ {env_id} 테스트 완료!")
    print(f"   📊 Episodes: {episode_count}, Successes: {success_count}")
    print(f"   ⏱️  Total steps: {steps}")
    
    return episode_count, success_count

def main():
    """메인 실행 함수"""
    print("🚀 Panda MuJoCo Gym - 모든 환경 테스트")
    print("=" * 70)
    print("💡 각 환경을 순서대로 시각화하면서 테스트합니다.")
    print("💡 Ctrl+C로 언제든 중단할 수 있습니다.")
    print("💡 각 환경 사이에서 Enter를 눌러 다음으로 진행합니다.\n")
    
    # 사용 가능한 모든 환경 출력
    print(f"📋 사용 가능한 환경들 ({len(panda_mujoco_gym.ENV_IDS)}개):")
    for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS, 1):
        print(f"  {i}. {env_id}")
    
    # 사용자 선택
    print("\n🎯 테스트 옵션:")
    print("  1. 모든 환경 자동 테스트")
    print("  2. 특정 환경만 선택해서 테스트")
    print("  3. 빠른 테스트 (각 환경 200스텝)")
    
    try:
        choice = input("\n선택하세요 (1/2/3, 기본값: 1): ").strip()
        
        if choice == "2":
            # 특정 환경 선택
            print("\n환경 번호를 입력하세요:")
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS, 1):
                print(f"  {i}. {env_id}")
            
            env_num = int(input("번호 입력: ")) - 1
            if 0 <= env_num < len(panda_mujoco_gym.ENV_IDS):
                selected_env = panda_mujoco_gym.ENV_IDS[env_num]
                test_single_env(selected_env, steps=1000, sleep_time=0.1)
            else:
                print("❌ 잘못된 번호입니다.")
                return
                
        elif choice == "3":
            # 빠른 테스트
            print("\n⚡ 빠른 테스트 모드 (각 환경 200스텝)")
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS):
                print(f"\n[{i+1}/{len(panda_mujoco_gym.ENV_IDS)}] {env_id}")
                test_single_env(env_id, steps=200, sleep_time=0.05)
                
                if i < len(panda_mujoco_gym.ENV_IDS) - 1:
                    input("\n⏸️  Press Enter to continue...")
        
        else:
            # 모든 환경 테스트 (기본값)
            print("\n🎬 모든 환경 테스트 시작!")
            
            results = []
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS):
                print(f"\n[{i+1}/{len(panda_mujoco_gym.ENV_IDS)}] {env_id}")
                episodes, successes = test_single_env(env_id, steps=500, sleep_time=0.1)
                results.append((env_id, episodes, successes))
                
                if i < len(panda_mujoco_gym.ENV_IDS) - 1:
                    input("\n⏸️  Press Enter to continue to next environment...")
            
            # 최종 결과 요약
            print("\n" + "=" * 70)
            print("📊 최종 테스트 결과 요약")
            print("=" * 70)
            for env_id, episodes, successes in results:
                print(f"{env_id:30} | Episodes: {episodes:2d} | Successes: {successes:3d}")
        
        print("\n🎉 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 사용자가 테스트를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
