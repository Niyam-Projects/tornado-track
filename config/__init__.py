"""Shared configuration loader using pydantic-settings + YAML."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


class DataConfig(BaseModel):
    root: str = r"E:\projects\tornado-track"
    dat_dir: str = r"E:\projects\tornado-track\dat"
    events_dir: str = r"E:\projects\tornado-track\events"
    checkpoints_dir: str = r"E:\projects\tornado-track\checkpoints"
    reports_dir: str = r"E:\projects\tornado-track\reports"
    index_path: str = r"E:\projects\tornado-track\index.parquet"
    stats_path: str = r"E:\projects\tornado-track\stats.json"


class DATConfig(BaseModel):
    base_url: str = "https://services.dat.noaa.gov/arcgis/rest/services/nws_damageassessmenttoolkit/DamageViewer/FeatureServer"
    years_back: int = 5
    event_type: str = "Tornado"


class MRMSConfig(BaseModel):
    bucket: str = "noaa-mrms-pds"
    region: str = "us-east-1"
    anonymous: bool = True
    variables: List[str] = [
        "ReflectivityAtLowestAltitude",
        "AzShear_0-2kmAGL",
        "AzShear_3-6kmAGL",
        "MESH",
        "RotationTrack30min",
        "RotationTrack60min",
        "RotationTrackML30min",
        "RotationTrackML60min",
    ]
    time_window_minutes: int = 90
    spatial_buffer_km: int = 50


class GRIBConfig(BaseModel):
    normalize_lon: bool = True


class ZarrConfig(BaseModel):
    grid_size: int = 200
    chunk_t: int = 10
    chunk_h: int = 64
    chunk_w: int = 64
    overlap: int = 8


class RewardConfig(BaseModel):
    w_track_proximity: float = 1.0
    w_rotation_anchor: float = 0.5
    w_false_active: float = 2.0
    w_lifecycle: float = 0.8


class LifecycleConfig(BaseModel):
    touchdown_threshold: float = 0.7
    lift_threshold: float = 0.3


class TrainingConfig(BaseModel):
    episodes_per_stage: int = 5000
    max_steps_per_episode: int = 90
    train_val_test_split: List[float] = [0.70, 0.15, 0.15]
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward: RewardConfig = RewardConfig()
    lifecycle: LifecycleConfig = LifecycleConfig()


class ModelConfig(BaseModel):
    cnn_channels: List[int] = [32, 64, 128]
    lstm_hidden: int = 256
    actor_hidden: int = 128
    lifecycle_hidden: int = 64
    ef_classification: bool = True
    ef_classes: int = 6


class InferenceConfig(BaseModel):
    confidence_sigma: List[float] = [1.0, 2.0]
    output_format: str = "geojson"


class CurriculumConfig(BaseModel):
    monster_threshold: float = 0.015  # max_rotation_score s⁻¹ → Tier 1
    weak_threshold: float = 0.005     # max_rotation_score s⁻¹ → Tier 3
    stage1_tier: int = 1              # Stage 1 uses only Tier 1 events
    stage2_tier: int = 2              # Stage 2 uses Tier 1+2 events
    stage3_tier: int = 3              # Stage 3 uses all tiers


class NormalizationConfig(BaseModel):
    rotation_clip_low: float = 0.010   # s⁻¹ → maps to 0.0
    rotation_clip_high: float = 0.040  # s⁻¹ → maps to 1.0


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TORNADO_")

    data: DataConfig = DataConfig()
    dat: DATConfig = DATConfig()
    mrms: MRMSConfig = MRMSConfig()
    grib: GRIBConfig = GRIBConfig()
    zarr: ZarrConfig = ZarrConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
    normalization: NormalizationConfig = NormalizationConfig()


def load_config(path: Path = _CONFIG_PATH) -> AppConfig:
    """Load config from YAML file, falling back to defaults."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return AppConfig.model_validate(raw)
    return AppConfig()


# Module-level singleton
cfg: AppConfig = load_config()
