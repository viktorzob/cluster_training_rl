"""
TD3 agent (Twin Delayed Deep Deterministic Policy Gradient).
Architecture mirrors Stable-Baselines3 conventions using PyTorch.

Optional normalisation:
  - LayerNorm on actor hidden layers  (use_layer_norm_actor=False by default)
  - BatchNorm on critic hidden layers (use_batch_norm_critic=False by default)
  - Running reward normalisation
  - Running observation normalisation
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.ptr        = 0
        self.size       = 0

        self.obs     = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),   dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ── Running normalizers ──────────────────────────────────────────────────────

class RunningMeanStd:
    """Welford online algorithm for mean/variance."""
    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape) if self.mean.shape else np.asarray(x, dtype=np.float64).reshape(-1)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_n    = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_n)

    def _update_from_moments(self, mean, var, n):
        delta      = mean - self.mean
        tot_count  = self.count + n
        new_mean   = self.mean + delta * n / tot_count
        m_a        = self.var  * self.count
        m_b        = var       * n
        M2         = m_a + m_b + delta ** 2 * self.count * n / tot_count
        self.mean  = new_mean
        self.var   = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        norm = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(norm, -clip, clip).astype(np.float32)


# ── Network building blocks ──────────────────────────────────────────────────

def _mlp_block(in_dim: int, out_dim: int, use_layer_norm: bool, use_batch_norm: bool):
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (256, 256),
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(_mlp_block(in_dim, h, use_layer_norm=use_layer_norm, use_batch_norm=False))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Twin critics (Q1 and Q2) in a single module."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (256, 256),
        use_batch_norm: bool = False,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = self._build(in_dim, hidden_sizes, use_batch_norm)
        self.q2 = self._build(in_dim, hidden_sizes, use_batch_norm)

    @staticmethod
    def _build(in_dim: int, hidden_sizes: tuple, use_batch_norm: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_sizes:
            layers.append(_mlp_block(d, h, use_layer_norm=False, use_batch_norm=use_batch_norm))
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x  = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


# ── TD3 Agent ────────────────────────────────────────────────────────────────

class TD3:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        # Architecture
        hidden_sizes: tuple = (256, 256),
        use_layer_norm_actor: bool  = False,
        use_batch_norm_critic: bool = False,
        # Normalisation
        normalize_observations: bool = True,
        normalize_rewards: bool      = True,
        # TD3 hyperparams
        gamma: float         = 0.99,
        tau: float           = 0.005,
        actor_lr: float      = 3e-4,
        critic_lr: float     = 3e-4,
        batch_size: int      = 256,
        buffer_size: int     = 1_000_000,
        learning_starts: int = 5_000,
        policy_delay: int    = 2,
        target_noise: float  = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.3,
        # Logging
        tensorboard_log: str | None = None,
        log_interval: int = 100,
        device: str = "auto",
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_delay    = policy_delay
        self.target_noise    = target_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.log_interval      = log_interval
        self.normalize_observations = normalize_observations
        self.normalize_rewards      = normalize_rewards

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        # Networks
        self.actor        = Actor(obs_dim, action_dim, hidden_sizes, use_layer_norm_actor).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic        = Critic(obs_dim, action_dim, hidden_sizes, use_batch_norm_critic).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        # Running normalizers
        self.obs_rms    = RunningMeanStd((obs_dim,))
        self.reward_rms = RunningMeanStd(())

        # Counters
        self.total_steps  = 0
        self.update_count = 0

        # TensorBoard
        self.writer: SummaryWriter | None = None
        if tensorboard_log:
            self.writer = SummaryWriter(log_dir=tensorboard_log)

        # Recent episode metrics for logging
        self._ep_rewards: deque = deque(maxlen=100)
        self._ep_profits: deque = deque(maxlen=100)
        self._cluster_profits: dict = {0: deque(maxlen=50), 1: deque(maxlen=50), 2: deque(maxlen=50)}

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_norm = self._normalize_obs(obs)
        obs_t    = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)
        self.actor.train()
        if not deterministic:
            noise  = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        obs_norm      = self._normalize_obs(obs,      update=True)
        next_obs_norm = self._normalize_obs(next_obs, update=False)
        rew_norm      = self._normalize_reward(reward)
        self.replay_buffer.add(obs_norm, action, rew_norm, next_obs_norm, float(done))
        self.total_steps += 1

    def train_step(self) -> dict | None:
        if len(self.replay_buffer) < self.learning_starts:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t  = self.critic_target(next_obs, next_actions)
            q_target     = rewards + self.gamma * (1 - dones) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor_target,  self.actor)
            self._soft_update(self.critic_target, self.critic)
            actor_loss_val = actor_loss.item()

        self.update_count += 1

        metrics = {"critic_loss": critic_loss.item()}
        if actor_loss_val is not None:
            metrics["actor_loss"] = actor_loss_val
        return metrics

    def log_episode(self, reward: float, info: dict):
        self._ep_rewards.append(reward)
        profit = info.get("total_profit", reward)
        self._ep_profits.append(profit)
        cluster = info.get("cluster", -1)
        if cluster in self._cluster_profits:
            self._cluster_profits[cluster].append(profit)

        if self.writer and self.total_steps % self.log_interval == 0:
            ep = len(self._ep_rewards)
            self.writer.add_scalar("rollout/ep_rew_mean",    np.mean(self._ep_rewards), self.total_steps)
            self.writer.add_scalar("rollout/profit_mean",    np.mean(self._ep_profits), self.total_steps)
            for c, q in self._cluster_profits.items():
                if q:
                    self.writer.add_scalar(f"rollout/profit_cluster{c}", np.mean(q), self.total_steps)

    def log_train_metrics(self, metrics: dict):
        if self.writer and metrics:
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, self.total_steps)

    def save(self, path: str):
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "actor_opt":      self.actor_optimizer.state_dict(),
            "critic_opt":     self.critic_optimizer.state_dict(),
            "obs_rms_mean":   self.obs_rms.mean,
            "obs_rms_var":    self.obs_rms.var,
            "obs_rms_count":  self.obs_rms.count,
            "rew_rms_mean":   self.reward_rms.mean,
            "rew_rms_var":    self.reward_rms.var,
            "rew_rms_count":  self.reward_rms.count,
            "total_steps":    self.total_steps,
            "update_count":   self.update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.obs_rms.mean   = ckpt["obs_rms_mean"]
        self.obs_rms.var    = ckpt["obs_rms_var"]
        self.obs_rms.count  = ckpt["obs_rms_count"]
        self.reward_rms.mean  = ckpt["rew_rms_mean"]
        self.reward_rms.var   = ckpt["rew_rms_var"]
        self.reward_rms.count = ckpt["rew_rms_count"]
        self.total_steps  = ckpt["total_steps"]
        self.update_count = ckpt["update_count"]

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _normalize_obs(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if not self.normalize_observations:
            return obs.astype(np.float32)
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _normalize_reward(self, reward: float) -> float:
        if not self.normalize_rewards:
            return reward
        self.reward_rms.update(np.array([reward]))
        # Clamp std from below so near-constant rewards don't blow up the scale.
        std = max(math.sqrt(float(self.reward_rms.var)), 1.0)
        return float(reward / std)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
