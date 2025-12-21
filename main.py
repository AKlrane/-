import os
from pathlib import Path
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import IndustryEnv
from config import load_config, Config


class EpisodeLengthWrapper(gym.Wrapper):
    """
    强制限制episode长度为n_steps的Wrapper。
    每个环境运行n_steps步后自动truncated，然后PPO会自动reset环境。
    这样可以确保每个rollout周期都是"新鲜"的环境状态，防止公司数量无限增长。
    """
    def __init__(self, env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        """重置环境并重置步数计数器"""
        self.current_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步，如果达到最大步数则truncated"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # 达到最大步数时强制truncated
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Wrapper to scale rewards by a factor to reduce value estimation error.
    This helps stabilize training by keeping value estimates in a reasonable range.
    """
    def __init__(self, env, scale: float = 0.1):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, reward):
        return reward * self.scale


class MapObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert company locations into a discrete grid map.
    Creates a [H, W, C=7] tensor where each channel represents the sum of log(capital) 
    for companies of that sector type in each grid cell.
    
    Optimized version: pre-computes normalization factors and uses vectorized operations.
    """
    def __init__(self, env, grid_size: int = None):
        super().__init__(env)
        # 获取底层IndustryEnv（可能需要unwrap多个wrapper）
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        self.env = env  # 保存包装后的环境（用于step等操作）
        self.base_env = base_env  # 保存底层IndustryEnv（用于访问属性）
        
        # 从底层环境获取属性，添加安全检查
        if not hasattr(base_env, 'size') or not hasattr(base_env, 'num_sectors'):
            raise ValueError(f"Expected IndustryEnv, but got {type(base_env)}. Cannot access size or num_sectors.")
        
        self.grid_size = int(base_env.size) if grid_size is None else grid_size
        self.num_sectors = base_env.num_sectors
        
        # Pre-compute normalization factors (only computed once)
        self.map_min = base_env.map_min
        self.map_max = base_env.map_max
        self.map_range = self.map_max - self.map_min
        
        # Observation space: grid map + tier channel
        # Shape: (grid_size, grid_size, num_sectors + 1)
        # Last channel contains current_tier information (0-5, normalized to [0, 1])
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.grid_size, self.grid_size, self.num_sectors + 1),
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Pre-allocate grid map with tier channel
        grid_map = np.zeros((self.grid_size, self.grid_size, self.num_sectors + 1), dtype=np.float32)
        
        # Get current tier from base environment and normalize to [0, 1]
        current_tier = self.base_env.current_tier_index
        tier_normalized = float(current_tier) / 5.0  # Normalize 0-5 to 0-1
        
        # Fill tier channel with normalized tier value
        grid_map[:, :, self.num_sectors] = tier_normalized
        
        # Early return if no companies (but tier channel is already filled)
        if len(self.base_env.companies) == 0:
            return grid_map
        
        # Vectorized computation: extract all company data at once
        companies = self.base_env.companies
        n_companies = len(companies)
        
        # Pre-allocate arrays for vectorized operations
        x_coords = np.array([c.location[0] for c in companies], dtype=np.float32)
        y_coords = np.array([c.location[1] for c in companies], dtype=np.float32)
        sector_ids = np.array([c.sector_id for c in companies], dtype=np.int32)
        capitals = np.array([c.capital for c in companies], dtype=np.float32)
        
        # Normalize coordinates (vectorized)
        x_norm = np.clip((x_coords - self.map_min) / self.map_range, 0.0, 1.0)
        y_norm = np.clip((y_coords - self.map_min) / self.map_range, 0.0, 1.0)
        
        # Convert to grid indices (vectorized)
        grid_x = np.clip((x_norm * self.grid_size).astype(np.int32), 0, self.grid_size - 1)
        grid_y = np.clip((y_norm * self.grid_size).astype(np.int32), 0, self.grid_size - 1)
        
        # Compute log1p of capitals (vectorized)
        capital_log = np.log1p(capitals)
        
        # Filter valid sector IDs
        valid_mask = (sector_ids >= 0) & (sector_ids < self.num_sectors)
        
        # Accumulate into grid map (CPU-optimized: use NumPy's add.at for vectorized accumulation)
        # This is much faster than Python loops on CPU (37x speedup)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_grid_y = grid_y[valid_indices]
            valid_grid_x = grid_x[valid_indices]
            valid_sector_ids = sector_ids[valid_indices]
            valid_capital_log = capital_log[valid_indices]
            
            # Use NumPy's add.at for vectorized accumulation (CPU-friendly)
            # This avoids Python loop overhead and leverages CPU vectorization
            np.add.at(grid_map, (valid_grid_y, valid_grid_x, valid_sector_ids), valid_capital_log)
        
        return grid_map


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
            h, w, c = observation_space.shape
        else:
            # Fallback to default values if observation_space is not as expected
            # This should match MapObservationWrapper's default grid_size (int(env.size) = 40)
            # Channels: 7 sectors + 1 tier channel = 8
            import warnings
            warnings.warn(f"Unexpected observation_space: {observation_space}. Using default (40, 40, 8).")
            h, w, c = 40, 40, 8
        
        self.num_channels = c
        
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        
        conv_output_size = 5 * 5 * 64
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 4 and observations.shape[-1] == self.num_channels:
            observations = observations.permute(0, 3, 1, 2)
        
        x = self.cnn(observations)
        x = self.fc(x)
        return x


def create_env(config: Config, seed=None, max_episode_steps=64):
    """
    创建环境，强制每个episode最多运行max_episode_steps步。
    
    Args:
        config: 配置对象
        seed: 随机种子（用于重置时的随机初始化）
        max_episode_steps: 每个episode的最大步数（默认64，与n_steps一致）
    """
    env = IndustryEnv(config.environment)
    if seed is not None:
        env.reset(seed=seed, options={"initial_firms": config.environment.initial_firms})
    
    # 强制限制episode长度为max_episode_steps
    # 这样每个rollout周期后环境会自动reset，重新随机初始化
    env = EpisodeLengthWrapper(env, max_episode_steps=max_episode_steps)
    
    # 应用观察wrapper
    # 注意：不再使用RewardScaleWrapper，因为env内部已经对利润部分乘以0.15
    # creation_reward和penalty保持原值，这样reward范围适合PPO学习
    env = MapObservationWrapper(env)
    return env


def train(config: Config):
    def make_env(i):
        def _init():
            # 每个环境使用不同的seed，确保重置时随机初始化不同
            # max_episode_steps=64，确保每个rollout周期后自动reset
            env = create_env(config, seed=config.training.seed + i, max_episode_steps=64)
            env = Monitor(env, filename=os.path.join(config.training.log_dir, f"env_{i}"))
            return env
        return _init
    
    # 使用SubprocVecEnv实现真正的并行环境（利用多核CPU）
    # DummyVecEnv是串行的，SubprocVecEnv是真正并行的
    num_envs = config.training.num_envs
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # 评估环境也使用相同的创建方式，但只需要单个环境
    # 使用SubprocVecEnv包装单个环境，与训练环境类型一致（避免warning）
    def make_eval_env():
        def _init():
            env = create_env(config, seed=config.training.seed + 1000, max_episode_steps=64)
            env = Monitor(env, filename=os.path.join(config.training.log_dir, "eval_env"))
            return env
        return _init
    
    eval_env = SubprocVecEnv([make_eval_env()])
    
    # 强制使用CPU（PPO在CPU上效率更高，特别是对于非CNN策略）
    device = "cpu"
    
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=nn.ReLU,
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.training.learning_rate,
        n_steps=64,  # 每个环境收集64步
        batch_size=64,  # 设置为与n_steps相同，确保每个rollout的数据在一个batch中训练
        gamma=config.training.gamma,
        n_epochs=5,  # 每次rollout后训练4个epoch
        verbose=1,
        tensorboard_log=config.training.log_dir,
        seed=config.training.seed,
        policy_kwargs=policy_kwargs,
        device=device,
        vf_coef=config.training.vf_coef,  # 使用配置中的vf_coef (0.05)
        ent_coef=config.training.ent_coef,  # 使用配置中的ent_coef
        clip_range_vf=None,  # 不clip value function，让value loss自然下降
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.save_freq // config.training.num_envs,
        save_path=config.training.checkpoint_dir,
        name_prefix="cnn_ppo_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.training.checkpoint_dir, "best_model"),
        log_path=config.training.log_dir,
        eval_freq=config.training.eval_freq,
        deterministic=True,
        render=False
    )
    
    rollout_steps = num_envs * 64  # 每次rollout的步数
    
    print(f"\n训练配置:")
    print(f"  环境数: {num_envs}, rollout: {rollout_steps}步, lr: {config.training.learning_rate:.6f}")
    print(f"  评估频率: 每{config.training.eval_freq}步, TensorBoard: {config.training.log_dir}\n")
    
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=1  # 每个iteration都记录训练指标到TensorBoard
    )
    
    final_path = os.path.join(config.training.checkpoint_dir, "cnn_ppo_final")
    model.save(final_path)
    
    vec_env.close()
    eval_env.close()
    return model


def main():
    config = load_config("config/config.json")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.training.log_dir = os.path.join(config.training.log_dir, f"cnn_ppo_{timestamp}")
    config.training.checkpoint_dir = os.path.join(config.training.checkpoint_dir, f"cnn_ppo_{timestamp}")
    
    Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    train(config)


if __name__ == "__main__":
    main()
