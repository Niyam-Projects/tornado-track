"""
Stage 2: Hunter Training

Spawns the agent 15 minutes before tornado touchdown (pre-tornadic environment).
Goal: learn to navigate toward intensifying RotationTrackML (mid-level) signatures.
Result: model learns to anticipate tornado initiation by watching mid-levels descend.

Initializes from Stage 1 checkpoint.

Usage:
    python -m training.stage2_hunter
"""
import logging
from pathlib import Path

import click

from config import cfg
from training.ppo_base import train_ppo

log = logging.getLogger(__name__)


@click.command()
@click.option("--episodes", default=None, type=int, help="Override episode count")
@click.option("--checkpoint-in", default=None, help="Override Stage 1 checkpoint path")
def main(episodes: int | None, checkpoint_in: str | None) -> None:
    """Run Stage 2 (Hunter) PPO training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_in = (
        Path(checkpoint_in)
        if checkpoint_in
        else Path(cfg.data.checkpoints_dir) / "stage1" / "checkpoint_final.pt"
    )

    log.info("=== Stage 2: Hunter Training ===")
    log.info("Initializing from: %s", ckpt_in)
    ckpt = train_ppo(
        stage=2,
        checkpoint_in=ckpt_in,
        total_episodes=episodes,
        run_name="stage2_hunter",
    )
    log.info("Stage 2 complete. Checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
