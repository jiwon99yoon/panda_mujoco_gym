# panda_mujoco_gym/my_test_env.py

import sys
import time
import gymnasium as gym
import panda_mujoco_gym
import os

def test_environment():
    """
    Franka Panda 로봇 환경을 테스트하는 함수
    """
    try:
        # 먼저 headless 모드로 테스트
        print("Testing environment in headless mode...")
        env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
        
        observation, info = env.reset()
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Dict observation의 각 키별 shape 출력
        if isinstance(observation, dict):
            print(f"Observation keys: {list(observation.keys())}")
            for key, value in observation.items():
                print(f"  {key} shape: {value.shape}")
        else:
            print(f"Initial observation shape: {observation.shape}")
        
        # 몇 스텝 실행해보기
        for step in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # terminated와 truncated를 boolean으로 변환
            terminated = bool(terminated)
            truncated = bool(truncated)
            
            print(f"Step {step}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                observation, info = env.reset()
                print("Episode reset!")
        
        env.close()
        print("Headless test completed successfully!")
        
        # 그래픽 모드 테스트
        print("\nTesting environment with graphics...")
        env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
        
        observation, info = env.reset()
        
        for step in range(100):  # 더 적은 스텝으로 테스트
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
            
            time.sleep(0.05)  # 더 빠른 실행
        
        env.close()
        print("Graphics test completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Trying alternative solutions...")
        
        # 대안: 가상 디스플레이 사용
        try:
            # 기존 X 서버 프로세스 정리
            os.system('pkill -f "Xvfb :99"')
            time.sleep(1)
            
            # 임시 파일 정리
            os.system('rm -f /tmp/.X99-lock')
            
            # 새 가상 디스플레이 시작
            os.environ['DISPLAY'] = ':99'
            os.system('Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &')
            time.sleep(2)  # 디스플레이 시작 대기
            
            env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
            observation, info = env.reset()
            
            for step in range(50):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                
                # boolean 변환
                terminated = bool(terminated)
                truncated = bool(truncated)
                
                if terminated or truncated:
                    observation, info = env.reset()
                    
                if step % 10 == 0:
                    print(f"Alternative step {step}: reward={reward:.4f}")
            
            env.close()
            print("Virtual display test completed!")
            
        except Exception as e2:
            print(f"Alternative solution also failed: {e2}")
            print("Please check your OpenGL installation and graphics drivers.")

if __name__ == "__main__":
    # 환경 변수 설정
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
    test_environment()

# import sys
# import time
# import gymnasium as gym
# import panda_mujoco_gym
# import os

# def test_environment():
#     """
#     Franka Panda 로봇 환경을 테스트하는 함수
#     """
#     try:
#         # 먼저 headless 모드로 테스트
#         print("Testing environment in headless mode...")
#         env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
        
#         observation, info = env.reset()
#         print(f"Environment created successfully!")
#         print(f"Observation space: {env.observation_space}")
#         print(f"Action space: {env.action_space}")
#         print(f"Initial observation shape: {observation.shape}")
        
#         # 몇 스텝 실행해보기
#         for step in range(10):
#             action = env.action_space.sample()
#             observation, reward, terminated, truncated, info = env.step(action)
#             print(f"Step {step}: reward={reward}, terminated={terminated}, truncated={truncated}")
            
#             if terminated or truncated:
#                 observation, info = env.reset()
#                 print("Episode reset!")
        
#         env.close()
#         print("Headless test completed successfully!")
        
#         # 그래픽 모드 테스트
#         print("\nTesting environment with graphics...")
#         env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
        
#         observation, info = env.reset()
        
#         for step in range(100):  # 더 적은 스텝으로 테스트
#             action = env.action_space.sample()
#             observation, reward, terminated, truncated, info = env.step(action)
            
#             if terminated or truncated:
#                 observation, info = env.reset()
            
#             time.sleep(0.05)  # 더 빠른 실행
        
#         env.close()
#         print("Graphics test completed successfully!")
        
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         print("Trying alternative solutions...")
        
#         # 대안: 가상 디스플레이 사용
#         try:
#             # xvfb 설치: sudo apt install xvfb
#             os.environ['DISPLAY'] = ':99'
#             os.system('Xvfb :99 -screen 0 1024x768x24 &')
            
#             env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
#             observation, info = env.reset()
            
#             for _ in range(50):
#                 action = env.action_space.sample()
#                 observation, reward, terminated, truncated, info = env.step(action)
                
#                 if terminated or truncated:
#                     observation, info = env.reset()
            
#             env.close()
#             print("Virtual display test completed!")
            
#         except Exception as e2:
#             print(f"Alternative solution also failed: {e2}")
#             print("Please check your OpenGL installation and graphics drivers.")

# if __name__ == "__main__":
#     # 환경 변수 설정
#     os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
#     os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
#     test_environment()

# import sys
# import time
# import gymnasium as gym
# import panda_mujoco_gym

# if __name__ == "__main__":
#     env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

#     observation, info = env.reset()

#     for _ in range(1000):
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)

#         if terminated or truncated:
#             observation, info = env.reset()

#         time.sleep(0.2)

#     env.close()

# import time
# import gymnasium as gym
# import panda_mujoco_gym

# env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
# obs, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
#     time.sleep(0.02)

# env.close()
