"""
Shared CleanRL-style PPO training loop for all three curriculum stages.

Each stage script calls `train_ppo(stage=N, ...)` which:
  1. Creates TornadoTrackEnv with the appropriate stage/spawn mode
  2. Runs the PPO update loop for `total_episodes` episodes
  3. Logs to TensorBoard
  4. Saves checkpoints to E:\\projects\\tornado-track\\checkpoints\\stage{N}\\

Reference: CleanRL PPO (https://github.com/vwxyzjn/cleanrl)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from env.tornado_env import TornadoTrackEnv
from model.policy import TornadoPolicy

log = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, n_steps: int, obs_shape: tuple, action_dim: int, device: torch.device):
        self.obs = torch.zeros((n_steps, *obs_shape), device=device)
        self.actions = torch.zeros((n_steps, action_dim), device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.ptr = 0
        self.n_steps = n_steps

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(rewards)
    advantages = torch.zeros(n, device=rewards.device)
    last_adv = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1].item()
        next_done = 1.0 - dones[t].item()
        delta = rewards[t] + gamma * next_val * next_done - values[t]
        last_adv = delta + gamma * gae_lambda * next_done * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_ppo(
    stage: int,
    checkpoint_in: Path | None = None,
    total_episodes: int | None = None,
    run_name: str | None = None,
) -> Path:
    """
    Run PPO training for a given curriculum stage.

    Args:
        stage:          Curriculum stage (1, 2, or 3).
        checkpoint_in:  Path to a prior-stage checkpoint to initialize from.
        total_episodes: Override number of episodes (default from config).
        run_name:       TensorBoard run name.

    Returns:
        Path to the final saved checkpoint.
    """
    tc = cfg.training
    total_episodes = total_episodes or tc.episodes_per_stage
    run_name = run_name or f"stage{stage}_{int(time.time())}"

    ckpt_dir = Path(cfg.data.checkpoints_dir) / f"stage{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.data.reports_dir) / "tensorboard" / run_name
    writer = SummaryWriter(str(log_dir))

    log.info("Stage %d training | device=%s | episodes=%d", stage, _DEVICE, total_episodes)

    env = TornadoTrackEnv(stage=stage, split="train")
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    policy = TornadoPolicy().to(_DEVICE)
    if checkpoint_in and checkpoint_in.exists():
        log.info("Loading checkpoint: %s", checkpoint_in)
        state = torch.load(checkpoint_in, map_location=_DEVICE)
        policy.load_state_dict(state["policy"])

    optimizer = optim.Adam(policy.parameters(), lr=tc.learning_rate, eps=1e-5)

    buf = RolloutBuffer(tc.n_steps, obs_shape, action_dim, _DEVICE)
    episode_rewards: list[float] = []
    episode_count = 0
    global_step = 0

    obs_np, _ = env.reset()
    obs = torch.from_numpy(obs_np).float().to(_DEVICE)
    lstm_h = lstm_c = None
    ep_reward = 0.0

    while episode_count < total_episodes:
        with torch.no_grad():
            lstm_state = (lstm_h, lstm_c) if lstm_h is not None else None
            action, log_prob, _, value, aux = policy.get_action_and_value(
                obs.unsqueeze(0), lstm_state=lstm_state
            )
            lstm_h, lstm_c = aux["lstm_h"], aux["lstm_c"]
            lifecycle_prob = aux["lifecycle_prob"].squeeze().item()

        env.set_lifecycle_prob(lifecycle_prob)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(
            action_np,
            env.action_space.low,
            env.action_space.high,
        )

        next_obs_np, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        ep_reward += reward
        global_step += 1

        buf.add(
            obs=obs,
            action=action.squeeze(0),
            log_prob=log_prob.squeeze(),
            reward=torch.tensor(reward, device=_DEVICE),
            done=torch.tensor(float(done), device=_DEVICE),
            value=value.squeeze(),
        )

        if done:
            episode_count += 1
            episode_rewards.append(ep_reward)
            writer.add_scalar("train/episode_reward", ep_reward, episode_count)
            if episode_count % 50 == 0:
                mean_r = np.mean(episode_rewards[-50:])
                log.info(
                    "Stage %d | ep=%d/%d | mean_reward(50)=%.3f",
                    stage, episode_count, total_episodes, mean_r,
                )
                writer.add_scalar("train/mean_reward_50", mean_r, episode_count)

            obs_np, _ = env.reset()
            obs = torch.from_numpy(obs_np).float().to(_DEVICE)
            lstm_h = lstm_c = None
            ep_reward = 0.0
        else:
            obs = torch.from_numpy(next_obs_np).float().to(_DEVICE)

        # PPO update when buffer is full
        if buf.is_full():
            with torch.no_grad():
                _, _, _, last_value, _ = policy.get_action_and_value(
                    obs.unsqueeze(0),
                    lstm_state=(lstm_h, lstm_c) if lstm_h is not None else None,
                )
            advantages, returns = compute_gae(
                buf.rewards, buf.values, buf.dones,
                last_value.item(), tc.gamma, tc.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten buffer for minibatch sampling
            b_obs = buf.obs
            b_actions = buf.actions
            b_log_probs = buf.log_probs
            b_returns = returns
            b_advantages = advantages

            for _ in range(tc.n_epochs):
                idxs = torch.randperm(tc.n_steps, device=_DEVICE)
                for start in range(0, tc.n_steps, tc.batch_size):
                    end = start + tc.batch_size
                    mb_idx = idxs[start:end]

                    _, new_log_prob, entropy, new_value, _ = policy.get_action_and_value(
                        b_obs[mb_idx], action=b_actions[mb_idx]
                    )

                    ratio = (new_log_prob - b_log_probs[mb_idx]).exp()
                    adv = b_advantages[mb_idx]

                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - tc.clip_coef, 1 + tc.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = ((new_value - b_returns[mb_idx]) ** 2).mean()
                    ent_loss = entropy.mean()

                    loss = pg_loss + tc.vf_coef * v_loss - tc.ent_coef * ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), tc.max_grad_norm)
                    optimizer.step()

            writer.add_scalar("train/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("train/value_loss", v_loss.item(), global_step)
            writer.add_scalar("train/entropy", ent_loss.item(), global_step)
            buf.reset()

    # Save final checkpoint
    ckpt_path = ckpt_dir / "checkpoint_final.pt"
    torch.save({"policy": policy.state_dict(), "stage": stage}, str(ckpt_path))
    log.info("Stage %d complete. Checkpoint → %s", stage, ckpt_path)
    writer.close()
    env.close()
    return ckpt_path
