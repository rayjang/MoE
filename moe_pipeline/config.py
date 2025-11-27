from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExpertSelectionConfig:
    """Configuration for deciding which experts to duplicate."""

    stats_path: Path
    metric: str = "usage"  # could be usage, entropy, loss_delta
    min_usage: float = 0.02
    min_entropy: float = 0.5
    num_new_experts: Optional[int] = None
    max_new_experts: int = 8
    tie_break_random: bool = False


@dataclass
class FsdpConfig:
    """Settings for FSDP/FSDP2 sharding."""

    use_fsdp2: bool = True
    wrap_granularity: str = "layer"
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    sync_module_states: bool = True


@dataclass
class TrainingPhaseConfig:
    """Defines a single training phase, e.g., freeze or unfreeze."""

    name: str
    epochs: int
    learning_rate: float
    freeze_existing_experts: bool = True
    unfreeze_layernorm: bool = False
    train_router_only: bool = False


@dataclass
class TrainingConfig:
    """High-level training configuration."""

    base_model_name: str = "OpenAccess-AI-Collective/GPT-OSS-20B"
    dataset_path: str = "data/science_moe_mix"
    output_dir: Path = Path("outputs/new_experts")
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    checkpoint_every: int = 500
    log_every: int = 50

    selection: ExpertSelectionConfig = field(
        default_factory=lambda: ExpertSelectionConfig(stats_path=Path("router_stats.json"))
    )
    fsdp: FsdpConfig = field(default_factory=FsdpConfig)
    phases: List[TrainingPhaseConfig] = field(
        default_factory=lambda: [
            TrainingPhaseConfig(
                name="freeze_old_train_new",
                epochs=1,
                learning_rate=5e-5,
                freeze_existing_experts=True,
                unfreeze_layernorm=True,
                train_router_only=False,
            ),
            TrainingPhaseConfig(
                name="router_alignment",
                epochs=1,
                learning_rate=2e-5,
                freeze_existing_experts=False,
                unfreeze_layernorm=True,
                train_router_only=True,
            ),
            TrainingPhaseConfig(
                name="progressive_unfreeze",
                epochs=1,
                learning_rate=1e-5,
                freeze_existing_experts=False,
                unfreeze_layernorm=True,
                train_router_only=False,
            ),
        ]
    )


@dataclass
class HubConfig:
    repo_id: str
    token: Optional[str] = None
    private: bool = True
