"""
Stage 3: Surveyor Training

Full time-series episodes from clear-sky through post-storm.
Goal: accurately signal tornado start and end (lifecycle head).
Penalty: heavy cost-of-effort penalty for every step the model claims
         a tornado is active when the DAT shows no damage.

Initializes from Stage 2 checkpoint.

Usage:
    python -m training.stage3_surveyor
"""
import logging
from pathlib import Path

import click

from config import cfg
from training.ppo_base import train_ppo

log = logging.getLogger(__name__)


@click.command()
@click.option("--episodes", default=None, type=int, help="Override episode count")
@click.option("--checkpoint-in", default=None, help="Override Stage 2 checkpoint path")
@click.option(
    "--tier", default=None, type=int,
    help="Max curriculum tier to train on (1=Monster only, 2=Moderate+, 3=all). "
         "Default: 3 (Stage 3 uses all events).",
)
def main(episodes: int | None, checkpoint_in: str | None, tier: int | None) -> None:
    """Run Stage 3 (Surveyor) PPO training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_in = (
        Path(checkpoint_in)
        if checkpoint_in
        else Path(cfg.data.checkpoints_dir) / "stage2" / "checkpoint_final.pt"
    )

    log.info("=== Stage 3: Surveyor Training ===")
    log.info("Initializing from: %s", ckpt_in)
    ckpt = train_ppo(
        stage=3,
        checkpoint_in=ckpt_in,
        total_episodes=episodes,
        run_name="stage3_surveyor",
        min_tier=tier,
    )
    log.info("Stage 3 complete. Checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
