"""Standalone evaluation script — run after training to generate sample outputs."""

import os
import sys
import torch
from transformers import PreTrainedTokenizerFast
from configs.model_config import ModelConfig
from models.model import BurkiLLM
from inference.generate import generate_text


EVAL_PROMPTS = [
    "What are you?",
    "Who made you?",
    "Tell me about yourself.",
    "What is Python?",
    "What is machine learning?",
    "Tell me a joke.",
    "What is a transformer?",
    "Introduce yourself.",
    "What can you do?",
    "What is deep learning?",
]


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/best_model.pt"
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/tokenizer"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"Loading model from {checkpoint}")
    model = BurkiLLM(config)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results = []
    for prompt in EVAL_PROMPTS:
        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=60, temperature=0.7, top_k=40, top_p=0.9,
        )
        line = f"Prompt: {prompt}\nResponse: {response}\n"
        results.append(line)
        print(f"\n{line}")

    out_path = "outputs/evaluation_results.txt"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("BurkiLLM Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(r + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
