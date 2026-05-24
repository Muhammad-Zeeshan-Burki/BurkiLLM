# BurkiLLM

A lightweight, educational decoder-only transformer language model developed by **Muhammad Zeeshan Burki**.

BurkiLLM is a portfolio-quality GPT-style language model with ~13M parameters. It trains on consumer hardware in under 2 hours and demonstrates core transformer concepts in clean, readable PyTorch code.

---

## Architecture

BurkiLLM uses a standard decoder-only transformer with causal self-attention:

| Component | Value |
|-----------|-------|
| Parameters | ~13M (10M non-embedding) |
| Layers | 6 |
| Hidden Size | 384 |
| Attention Heads | 6 |
| Head Dimension | 64 |
| Feed-Forward Size | 1536 (4× hidden) |
| Context Length | 256 tokens |
| Vocabulary | ~8192 (BPE) |
| Position Encoding | Learned |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Embeddings | Tied (input/output) |

### Design Choices

- **Pre-normalization** (LayerNorm before each sub-layer) for stable training
- **Causal masking** ensures autoregressive generation
- **Weight tying** between input embeddings and LM head reduces parameter count
- **Scaled dot-product attention** with learned positional embeddings
- **GPT-2 style residual scaling** for deep network stability

## Datasets

Total data size: well under 2 GB.

| Dataset | Size | Purpose |
|---------|------|---------| 
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 80k stories (subsampled) | Language modeling fundamentals |
| [WikiText-2](https://huggingface.co/datasets/wikitext) | ~36k lines | General language patterns |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | 5k examples (subsampled) | Basic instruction following |
| Custom identity data | 82 examples | Model self-identification |

## Project Structure

```
BurkiLLM/
├── configs/
│   └── model_config.py         # Model and training dataclasses
├── models/
│   └── model.py                # Transformer implementation
├── data/
│   ├── tokenizer.py            # BPE tokenizer training
│   ├── dataset.py              # Dataset classes
│   └── identity.json           # Identity instruction data
├── training/
│   └── trainer.py              # Training loop
├── inference/
│   └── generate.py             # Text generation
├── scripts/
│   ├── prepare_data.py         # Dataset download
│   ├── train_tokenizer.py      # Standalone tokenizer training
│   └── evaluate.py             # Evaluation and sample generation
├── train.py                    # Main entry point (runs everything)
├── chat.py                     # Interactive chat CLI
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support (recommended)
- NVIDIA GPU with 6GB+ VRAM, or CPU (slower)

### Installation

```bash
git clone https://github.com/yourusername/BurkiLLM.git
cd BurkiLLM
pip install -r requirements.txt
```

## Training

Run the full end-to-end pipeline:

```bash
python train.py
```

This executes six steps automatically:

1. **Download datasets** — TinyStories (80k), WikiText-2, Alpaca (5k)
2. **Train BPE tokenizer** — 8192 vocab on combined corpus
3. **Build model** — 6-layer transformer, ~13M parameters
4. **Pre-train** — Causal language modeling on text corpora (3 epochs)
5. **Fine-tune** — Instruction tuning with identity/Q&A data (20 epochs)
6. **Evaluate** — Generate sample outputs and save results

### Training Configuration

Edit `configs/model_config.py` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Samples per training batch |
| `learning_rate` | 3e-4 | Peak learning rate (with warmup) |
| `num_epochs` | 3 | Pre-training epochs |
| `weight_decay` | 0.01 | AdamW regularization |
| `mixed_precision` | True | FP16 training on CUDA |
| `tinystories_subset` | 80,000 | TinyStories sample size |
| `alpaca_subset` | 5,000 | Alpaca sample size |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8+ GB (RTX 40-series) |
| System RAM | 16 GB | 32 GB DDR5 |
| Storage | 10 GB | 20 GB |
| Training Time | — | ~1–2 hours on GPU |

## Inference

### Interactive Chat

```bash
python chat.py
```

### Python API

```python
import torch
from transformers import PreTrainedTokenizerFast
from configs.model_config import ModelConfig
from models.model import BurkiLLM
from inference.generate import generate_text

tokenizer = PreTrainedTokenizerFast.from_pretrained("outputs/tokenizer")
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model = BurkiLLM(config)

ckpt = torch.load("outputs/checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(generate_text(model, tokenizer, "Who are you?", max_new_tokens=80))
```

### Standalone Evaluation

```bash
python scripts/evaluate.py
python scripts/evaluate.py outputs/checkpoints/best_model.pt outputs/tokenizer
```

## Sample Outputs

After training, the model produces responses like:

```
You: Hello!
BurkiLLM: Hello! I am BurkiLLM, a lightweight language model by Muhammad Zeeshan Burki. How can I help you?

You: Who are you?
BurkiLLM: I am BurkiLLM, a lightweight transformer language model developed by Muhammad Zeeshan Burki.

You: What is Python?
BurkiLLM: Python is a high-level programming language known for its readability and versatility.

You: Tell me a joke.
BurkiLLM: Why do programmers prefer dark mode? Because light attracts bugs!
```

> **Note:** Actual outputs depend on training dynamics and random sampling. A ~13M parameter model has inherent limitations — responses may be repetitive or partially incoherent for complex queries.

## Limitations

- **Short context** (256 tokens) — cannot handle long documents
- **Small knowledge base** — limited by training data and parameter count
- **No factual grounding** — may generate incorrect information
- **English only** — trained exclusively on English text
- **Simple responses** — not suitable for complex reasoning or multi-turn dialogue
- **Educational project** — not designed for production deployment

## Future Improvements

- [ ] Rotary Position Embeddings (RoPE)
- [ ] KV caching for faster inference
- [ ] Flash Attention support
- [ ] Extended context length (512+)
- [ ] LoRA fine-tuning support
- [ ] Streaming text generation
- [ ] Larger instruction tuning datasets

## License

MIT

## Author

**Muhammad Zeeshan Burki** — Machine Learning Engineer

---

*Built from scratch with PyTorch.*
