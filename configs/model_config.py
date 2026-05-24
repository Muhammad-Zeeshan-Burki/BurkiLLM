from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    max_position_embeddings: int = 256   # increased from 128 for better context
    intermediate_size: int = 1536
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TrainingConfig:
    # ── Pre-training ──────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 3          # 3 pre-train epochs is plenty
    num_warmup_steps: int = 300
    grad_clip: float = 1.0
    patience: int = 3
    mixed_precision: bool = True
    max_seq_length: int = 256    # match max_position_embeddings

    # ── Data ──────────────────────────────────────────────────────
    tokenizer_vocab_size: int = 8192
    tinystories_subset: int = 80_000    # ~1 hr on GPU
    alpaca_subset: int = 5_000

    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
