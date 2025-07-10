#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) 기반 Panda 로봇 학습 코드
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import matplotlib.pyplot as plt
import json
import csv

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 사용자 정의 환경 등록
import panda_mujoco_gym

print("🤖 SAC 기반 Panda 로봇 학습 시작!")
print("=" * 60)

# 하이퍼파라미터 설정
class Config:
    # 환경 설정
    env_name = "FrankaSlideDense-v0"  # 학습할 환경 (Dense 보상으로 시작)
    max_episodes = 1000
    max_steps_per_episode = 200
    
    # SAC 하이퍼파라미터
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    gamma = 0.99
    tau = 0.005  # Soft update parameter
    alpha = 0.2  # Entropy regularization coefficient
    auto_entropy_tuning = True  # 자동 엔트로피 튜닝
    
    # 학습 설정
    batch_size = 256
    buffer_size = 1000000
    learning_starts = 1000
    train_freq = 1
    gradient_steps = 1
    
    # 평가 및 저장
    eval_freq = 50
    save_freq = 100
    log_freq = 10
    
    # 동영상 녹화 설정
    record_video = True
    video_freq = 100  # 몇 에피소드마다 동영상 녹화
    
    # 저장 경로 설정
    save_dir = "training_results"
    model_dir = "models"
    video_dir = "videos"
    log_dir = "logs"
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# 경험 재현 버퍼
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size

# 액터 네트워크 (정책 네트워크)
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

# 크리틱 네트워크 (Q-함수)
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 네트워크
        self.q1_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_head = nn.Linear(hidden_dim, 1)
        
        # Q2 네트워크 (Double Q-learning)
        self.q2_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_head(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_head(q2)
        
        return q1, q2

# SAC 에이전트
class SAC:
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.device = config.device
        
        # 네트워크 초기화
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # 자동 엔트로피 튜닝
        if config.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(config.alpha).to(self.device)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.config.gamma * min_qf_next_target
        
        # 크리틱 업데이트
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
        
        # 액터 업데이트
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # 엔트로피 계수 업데이트
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 타겟 네트워크 소프트 업데이트
        self.soft_update(self.critic_target, self.critic, self.config.tau)
        
        return {
            'critic_loss': qf_loss.item(),
            'actor_loss': policy_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# 평가 함수 (동영상 녹화 포함)
def evaluate_policy(env, agent, n_eval_episodes=10, record_video=False, video_dir=None, episode_num=None):
    avg_reward = 0.0
    success_count = 0
    
    # 동영상 녹화 환경 설정
    if record_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
        eval_env = RecordVideo(
            env, 
            video_folder=video_dir,
            name_prefix=f"eval_episode_{episode_num}",
            episode_trigger=lambda x: x == 0  # 첫 번째 에피소드만 녹화
        )
    else:
        eval_env = env
    
    for i in range(n_eval_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 성공 체크
            if info.get('is_success', False):
                success_count += 1
                break
        
        avg_reward += episode_reward
        
        # 첫 번째 에피소드만 녹화하므로 break
        if record_video and i == 0:
            break
    
    avg_reward /= n_eval_episodes
    success_rate = success_count / n_eval_episodes
    
    if record_video and video_dir:
        eval_env.close()
    
    return avg_reward, success_rate

# 메인 학습 함수
def train():
    print(f"🎯 환경: {config.env_name}")
    print(f"🖥️  디바이스: {config.device}")
    print(f"📊 최대 에피소드: {config.max_episodes}")
    print("-" * 60)
    
    # 저장 디렉토리 생성
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    if config.record_video:
        os.makedirs(config.video_dir, exist_ok=True)
    
    # 환경 초기화
    env = gym.make(config.env_name, render_mode=None)
    env = RecordEpisodeStatistics(env)
    
    eval_env = gym.make(config.env_name, render_mode="rgb_array")
    
    # 차원 정보
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"📏 관찰 차원: {obs_dim}")
    print(f"🎮 행동 차원: {action_dim}")
    print(f"🎯 행동 범위: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")
    
    # 에이전트 및 버퍼 초기화
    agent = SAC(obs_dim, action_dim, config)
    replay_buffer = ReplayBuffer(config.buffer_size, obs_dim, action_dim, config.device)
    
    # 로그 저장용
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_success_rates = []
    training_log = []
    
    # CSV 로그 파일 초기화
    log_file = os.path.join(config.log_dir, f"training_log_{config.env_name}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Length', 'Avg_Reward', 'Eval_Reward', 'Success_Rate'])
    
    # 학습 시작
    print("\n🚀 학습 시작!")
    print("=" * 60)
    
    state, _ = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_num = 0
    
    for total_step in range(config.max_episodes * config.max_steps_per_episode):
        # 행동 선택
        if total_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        # 환경 스텝
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 버퍼에 저장
        replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_step += 1
        
        # 에피소드 종료 처리
        if done or episode_step >= config.max_steps_per_episode:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            episode_num += 1
            
            # 평균 보상 계산
            avg_reward = np.mean(episode_rewards[-config.log_freq:]) if len(episode_rewards) >= config.log_freq else np.mean(episode_rewards)
            
            # 로그 출력
            if episode_num % config.log_freq == 0:
                print(f"Episode {episode_num:4d} | Reward: {episode_reward:7.2f} | Avg Reward: {avg_reward:7.2f} | Steps: {episode_step:3d}")
            
            # 환경 리셋
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            
            # 최대 에피소드 도달 시 종료
            if episode_num >= config.max_episodes:
                break
        
        # 학습 업데이트
        if total_step >= config.learning_starts and total_step % config.train_freq == 0:
            if len(replay_buffer) >= config.batch_size:
                for _ in range(config.gradient_steps):
                    losses = agent.update_parameters(replay_buffer, config.batch_size)
        
        # 평가 및 동영상 녹화
        if total_step > 0 and total_step % (config.eval_freq * config.max_steps_per_episode) == 0:
            # 동영상 녹화 여부 결정
            should_record = config.record_video and (episode_num % config.video_freq == 0)
            
            avg_reward, success_rate = evaluate_policy(
                eval_env, agent, 
                record_video=should_record,
                video_dir=config.video_dir if should_record else None,
                episode_num=episode_num
            )
            
            eval_rewards.append(avg_reward)
            eval_success_rates.append(success_rate)
            
            print(f"🔍 평가 (Step {total_step}): Reward={avg_reward:.2f}, Success Rate={success_rate:.2f}")
            if should_record:
                print(f"🎥 동영상 저장: {config.video_dir}/eval_episode_{episode_num}*.mp4")
            
            # CSV 로그 업데이트
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                current_avg = np.mean(episode_rewards[-config.log_freq:]) if len(episode_rewards) >= config.log_freq else np.mean(episode_rewards)
                writer.writerow([episode_num, episode_rewards[-1] if episode_rewards else 0, 
                               episode_lengths[-1] if episode_lengths else 0, current_avg, avg_reward, success_rate])
        
        # 모델 저장
        if total_step > 0 and total_step % (config.save_freq * config.max_steps_per_episode) == 0:
            model_path = os.path.join(config.model_dir, f"sac_{config.env_name}_{total_step}.pth")
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'episode': episode_num,
                'total_step': total_step,
                'config': config.__dict__
            }, model_path)
            print(f"💾 모델 저장: {model_path}")
    
    print("\n✅ 학습 완료!")
    print(f"📊 총 에피소드: {len(episode_rewards)}")
    print(f"🎯 평균 보상: {np.mean(episode_rewards):.2f}")
    print(f"🏆 최고 보상: {np.max(episode_rewards):.2f}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(config.model_dir, f"sac_{config.env_name}_final.pth")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'episode': episode_num,
        'total_step': total_step,
        'config': config.__dict__
    }, final_model_path)
    
    # 학습 통계 JSON 저장
    training_stats = {
        'config': config.__dict__,
        'total_episodes': len(episode_rewards),
        'total_steps': total_step,
        'final_avg_reward': np.mean(episode_rewards),
        'best_reward': np.max(episode_rewards),
        'final_success_rate': eval_success_rates[-1] if eval_success_rates else 0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates
    }
    
    stats_path = os.path.join(config.save_dir, f"training_stats_{config.env_name}.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # 학습 결과 시각화
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # 이동 평균 계산
    if len(episode_rewards) > 50:
        moving_avg = []
        window = 50
        for i in range(window, len(episode_rewards)):
            moving_avg.append(np.mean(episode_rewards[i-window:i]))
        
        plt.subplot(2, 3, 3)
        plt.plot(moving_avg)
        plt.title('Moving Average Rewards (50 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    if eval_rewards:
        plt.subplot(2, 3, 4)
        plt.plot(eval_rewards)
        plt.title('Evaluation Rewards')
        plt.xlabel('Evaluation')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(eval_success_rates)
        plt.title('Success Rate')
        plt.xlabel('Evaluation')
        plt.ylabel('Success Rate')
        plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.hist(episode_rewards, bins=50, alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 그래프 저장
    plot_path = os.path.join(config.save_dir, f'training_results_{config.env_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📁 저장된 파일들:")
    print(f"   📊 모델: {config.model_dir}/")
    print(f"   📈 그래프: {plot_path}")
    print(f"   📋 로그: {log_file}")
    print(f"   📊 통계: {stats_path}")
    if config.record_video:
        print(f"   🎥 동영상: {config.video_dir}/")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
