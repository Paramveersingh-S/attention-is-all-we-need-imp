from dataclasses import dataclass, field
import torch

@dataclass
class ModelConfig:
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 4
    n_layers: int = 3
    vocab_size: int = 16000
    max_seq_len: int = 256
    dropout: float = 0.1
    label_smoothing: float = 0.1
    tie_embeddings: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_epochs: int = 20
    warmup_steps: int = 4000
    learning_rate_scale: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_every_steps: int = 50
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
