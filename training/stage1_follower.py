"""
Stage 1: Follower Training

Spawns the agent directly on the known DAT track (t=0).
Goal: learn that high RotationTrack30min + high Reflectivity = reward.
Result: model learns the physics of following an active tornado track.

Usage:
    python -m training.stage1_follower
"""
import logging

import click

from training.ppo_base import train_ppo

log = logging.getLogger(__name__)


@click.command()
@click.option("--episodes", default=None, type=int, help="Override episode count")
@click.option(
    "--tier", default=None, type=int,
    help="Max curriculum tier to train on (1=Monster only, 2=Moderate+, 3=all). "
         "Default: 1 (Stage 1 trains only on high-signal 'Monster' events).",
)
def main(episodes: int | None, tier: int | None) -> None:
    """Run Stage 1 (Follower) PPO training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Stage 1: Follower Training ===")
    ckpt = train_ppo(stage=1, total_episodes=episodes, run_name="stage1_follower", min_tier=tier)
    log.info("Stage 1 complete. Checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
