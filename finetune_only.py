"""
finetune_only.py
Re-runs ONLY the instruction fine-tuning step on the existing pre-trained checkpoint.
Use this after fixing identity.json without needing to redo hours of pre-training.
"""
import os
import random
import torch
from transformers import PreTrainedTokenizerFast
from configs.model_config import ModelConfig, TrainingConfig
from models.model import BurkiLLM
from data.dataset import InstructionDataset
from training.trainer import Trainer
from inference.generate import generate_text

CHECKPOINT  = "outputs/checkpoints/best_model.pt"
TOKENIZER   = "outputs/tokenizer"
SAVE_DIR    = "outputs/checkpoints"


def load_pretrained(checkpoint_path, tokenizer_path, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    config    = ModelConfig(
        vocab_size     = tokenizer.vocab_size,
        pad_token_id   = tokenizer.pad_token_id,
        bos_token_id   = tokenizer.bos_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    model = BurkiLLM(config)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model, tokenizer, config


def run_finetune(model, tokenizer, config, device):
    identity_data = InstructionDataset.load_identity_data("data/identity.json")
    augmented     = identity_data * 30
    random.shuffle(augmented)

    dataset = InstructionDataset(tokenizer, augmented, max_length=config.max_position_embeddings)
    split   = int(len(dataset) * 0.9)
    train_  = torch.utils.data.Subset(dataset, range(split))
    val_    = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    print(f"Fine-tune train: {len(train_)}, val: {len(val_)}")

    ft_cfg = TrainingConfig(
        batch_size       = 32,
        learning_rate    = 3e-5,
        weight_decay     = 0.01,
        num_epochs       = 25,
        num_warmup_steps = 30,
        grad_clip        = 1.0,
        patience         = 10,
        mixed_precision  = True,
        max_seq_length   = config.max_position_embeddings,
        device           = device,
    )

    trainer = Trainer(model, train_, val_, ft_cfg)
    trainer.train(save_dir=SAVE_DIR)
    return model


def run_inference(model, tokenizer):
    prompts = [
        "Hello!",
        "Who are you?",
        "What is AI?",
        "What is machine learning?",
        "What is Python?",
        "Tell me a joke.",
        "Introduce yourself.",
        "What is a transformer?",
        "What can you do?",
    ]
    print("\n" + "=" * 60)
    print("Inference test (greedy / low temperature)")
    print("=" * 60)
    results = []
    for p in prompts:
        r = generate_text(model, tokenizer, p, max_new_tokens=80, temperature=0.3, top_k=10, top_p=0.95)
        print(f"\nYou:      {p}")
        print(f"BurkiLLM: {r}")
        results.append(f"Prompt: {p}\nResponse: {r}\n")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inference_results.txt", "w", encoding="utf-8") as f:
        f.write("BurkiLLM Inference Test Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(r + "\n")
    print(f"\nSaved to outputs/inference_results.txt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, tokenizer, config = load_pretrained(CHECKPOINT, TOKENIZER, device)
    model = run_finetune(model, tokenizer, config, device)
    run_inference(model, tokenizer)
    print("\nDone. Run `python chat.py` to chat.")
