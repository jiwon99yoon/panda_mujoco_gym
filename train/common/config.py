#/home/dyros/panda_mujoco_gym/train/common/config.py
"""
학습 설정 클래스
"""

import os
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BaseConfig:
    """기본 설정 클래스"""
    # 환경 설정
    env_name: str = "FrankaSlideDense-v0"
    algorithm: str = "SAC"
    
    # 학습 설정
    total_timesteps: int = 1_000_000
    
    # 환경 개선 설정
    normalize_env: bool = True
    reward_scale: float = 0.1
    
    # 평가 설정
    eval_freq: int = 20_000
    n_eval_episodes: int = 20
    eval_deterministic: bool = True
    
    # 저장 설정
    save_freq: int = 100_000
    checkpoint_freq: int = 50_000
    
    # 디렉토리 설정
    base_dir: str = "outputs"
    experiment_name: Optional[str] = None
    
    # 난수 시드 (worker마다 seed+rank 로 분리)
    seed: int = 0
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.experiment_name is None:
            self.experiment_name = f"{self.env_name}_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 전역 시드 고정 (numpy, torch, random)
        import random, numpy as _np, torch as _th
        random.seed(self.seed)
        _np.random.seed(self.seed)
        _th.manual_seed(self.seed)
        
        # 실험별 디렉토리 설정
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        dirs_to_create = [
            self.exp_dir,
            self.model_dir,
            self.log_dir,
            self.checkpoint_dir,
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 디렉토리 생성: {dir_path}")


@dataclass
class SACConfig(BaseConfig):
    """SAC 전용 설정"""
    # 벡터 환경 병렬 개수
    n_envs: int = 4
    # (seed 는 BaseConfig 에서 상속받습니다)
        
    # SAC 하이퍼파라미터
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 512
    tau: float = 0.01
    gamma: float = 0.98
    train_freq: int = 4
    gradient_steps: int = 4
    
    # 네트워크 구조
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256, 128],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False
    })
    
    # 탐험 노이즈
    action_noise_std: float = 0.2
    
    # 추가 학습 기법
    use_sde: bool = False #SDE 비활성화
    sde_sample_freq: int = 4
    
    # 학습 단계 정의 (비디오 녹화용)
    stages: Dict[str, float] = field(default_factory=lambda: {
        '0_random': 0,
        '1_20percent': 0.2,
        '2_40percent': 0.4,
        '3_60percent': 0.6,
        '4_80percent': 0.8,
        '5_100percent': 1.0
    })
    
    def get_stage_timesteps(self) -> Dict[str, int]:
        """각 단계별 타임스텝 계산"""
        return {name: int(ratio * self.total_timesteps) 
                for name, ratio in self.stages.items()}
