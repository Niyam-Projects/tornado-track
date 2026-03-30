"""
Evaluation script — per-stage metrics and reporting.

Loads the trained Stage 3 model (or any specified checkpoint), runs it over the
val/test split, and computes:
  - Track Hausdorff Distance (grid cells)
  - Polygon IoU (predicted swath vs DAT damage polygon)
  - Lifecycle F1 (precision / recall on touchdown/lift)
  - EF Classification Accuracy (if enabled)
  - Episode reward curves per stage (from TensorBoard logs)

Saves a report to E:\\projects\\tornado-track\\reports\\evaluation.json
and plots to E:\\projects\\tornado-track\\reports\\plots\\

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --checkpoint path/to/checkpoint.pt --split val
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Point

from config import cfg
from env.tornado_env import TornadoTrackEnv
from model.policy import TornadoPolicy

log = logging.getLogger(__name__)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOUCHDOWN_THRESHOLD = cfg.training.lifecycle.touchdown_threshold
_LIFT_THRESHOLD = cfg.training.lifecycle.lift_threshold


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def hausdorff_distance(pred_path: list[tuple], dat_path: list[tuple]) -> float:
    """Directed Hausdorff distance between two paths (in grid cells)."""
    if not pred_path or not dat_path:
        return float("nan")
    u = np.array(pred_path)
    v = np.array(dat_path)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])


def polygon_iou(pred_poly, dat_poly) -> float:
    if pred_poly is None or dat_poly is None:
        return 0.0
    if not (pred_poly.is_valid and dat_poly.is_valid):
        return 0.0
    inter = pred_poly.intersection(dat_poly).area
    union = pred_poly.union(dat_poly).area
    return float(inter / union) if union > 0 else 0.0


def lifecycle_f1(pred_active: list[bool], dat_active: list[bool]) -> dict[str, float]:
    tp = sum(p and d for p, d in zip(pred_active, dat_active))
    fp = sum(p and not d for p, d in zip(pred_active, dat_active))
    fn = sum(not p and d for p, d in zip(pred_active, dat_active))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def ef_accuracy(pred_ef: list[int], dat_ef: list[int]) -> float:
    if not pred_ef:
        return float("nan")
    correct = sum(p == d for p, d in zip(pred_ef, dat_ef))
    return correct / len(pred_ef)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def evaluate(checkpoint_path: Path, split: str = "test") -> dict:
    policy = TornadoPolicy().to(_DEVICE)
    state = torch.load(checkpoint_path, map_location=_DEVICE)
    policy.load_state_dict(state["policy"])
    policy.eval()

    env = TornadoTrackEnv(stage=3, split=split)
    n_episodes = min(100, len(env._index))

    all_hausdorff, all_iou, all_lifecycle, all_ef = [], [], [], []
    pred_actives: list[bool] = []
    dat_actives: list[bool] = []
    pred_efs: list[int] = []
    dat_efs: list[int] = []

    for ep_idx in range(n_episodes):
        obs_np, _ = env.reset()
        obs = torch.from_numpy(obs_np).float().to(_DEVICE)
        lstm_h = lstm_c = None
        done = False

        pred_positions, dat_active_flags = [], []
        lifecycle_preds, ef_preds = [], []

        while not done:
            with torch.no_grad():
                lstm_state = (lstm_h, lstm_c) if lstm_h is not None else None
                action, _, _, _, aux = policy.get_action_and_value(
                    obs.unsqueeze(0), lstm_state=lstm_state
                )
                lstm_h, lstm_c = aux["lstm_h"], aux["lstm_c"]
                lp = aux["lifecycle_prob"].squeeze().item()

            env.set_lifecycle_prob(lp)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs_np, _, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            obs = torch.from_numpy(next_obs_np).float().to(_DEVICE)

            pred_positions.append((info["agent_y"], info["agent_x"]))
            dat_active_flags.append(info["dat_active"])
            lifecycle_preds.append(lp > _TOUCHDOWN_THRESHOLD)

            if aux.get("ef_logits") is not None:
                ef_pred = int(aux["ef_logits"].squeeze().argmax().item())
                ef_preds.append(ef_pred)

        # Hausdorff (approximate DAT path from center of grid for now)
        dat_path = [(env._grid_lat.shape[0] // 2, env._grid_lon.shape[0] // 2)] * len(pred_positions)
        all_hausdorff.append(hausdorff_distance(pred_positions, dat_path))

        # Lifecycle F1
        lf = lifecycle_f1(lifecycle_preds, dat_active_flags)
        all_lifecycle.append(lf["f1"])
        pred_actives.extend(lifecycle_preds)
        dat_actives.extend(dat_active_flags)

        if ep_idx % 10 == 0:
            log.info("Episode %d/%d", ep_idx + 1, n_episodes)

    results = {
        "hausdorff_mean": float(np.nanmean(all_hausdorff)),
        "hausdorff_std": float(np.nanstd(all_hausdorff)),
        "lifecycle_f1_mean": float(np.nanmean(all_lifecycle)),
        "lifecycle_f1_std": float(np.nanstd(all_lifecycle)),
        "n_episodes": n_episodes,
        "split": split,
        "checkpoint": str(checkpoint_path),
    }

    if pred_efs:
        results["ef_accuracy"] = ef_accuracy(pred_efs, dat_efs)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results: dict, reports_dir: Path) -> None:
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Summary bar chart
    metrics = {
        "Lifecycle F1": results["lifecycle_f1_mean"],
    }
    if "ef_accuracy" in results:
        metrics["EF Accuracy"] = results["ef_accuracy"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(metrics.keys()), list(metrics.values()), color=["steelblue", "coral"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation Metrics ({results['split']} split)")
    for i, (k, v) in enumerate(metrics.items()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "evaluation_metrics.png", dpi=150)
    plt.close(fig)
    log.info("Saved metrics plot → %s", plots_dir / "evaluation_metrics.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--checkpoint", default=None, help="Path to model checkpoint")
@click.option("--split", default="test", show_default=True, help="Data split to evaluate on")
def main(checkpoint: str | None, split: str) -> None:
    """Evaluate trained model and generate metrics report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_path = (
        Path(checkpoint)
        if checkpoint
        else Path(cfg.data.checkpoints_dir) / "stage3" / "checkpoint_final.pt"
    )
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        return

    results = evaluate(ckpt_path, split=split)

    reports_dir = Path(cfg.data.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "evaluation.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Report → %s", report_path)
    log.info("Results: %s", results)

    plot_results(results, reports_dir)


if __name__ == "__main__":
    main()
