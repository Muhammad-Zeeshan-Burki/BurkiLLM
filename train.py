import os
import sys
import random
import torch
from datasets import load_dataset
from configs.model_config import ModelConfig, TrainingConfig
from models.model import BurkiLLM
from data.tokenizer import train_tokenizer
from data.dataset import TextDataset, InstructionDataset
from training.trainer import Trainer
from inference.generate import generate_text


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    print("CUDA not available — using CPU (training will be slow)")
    return "cpu"


def load_raw_texts(config):
    """Download and subsample training corpora. Returns list of text strings."""
    print("\nDownloading TinyStories...")
    tinystories = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    ts_texts = [item["text"] for item in tinystories if item["text"].strip()]

    if len(ts_texts) > config.tinystories_subset:
        random.seed(42)
        ts_texts = random.sample(ts_texts, config.tinystories_subset)
    print(f"  TinyStories: {len(ts_texts)} examples (subsampled)")

    print("Downloading WikiText-2...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    wiki_texts = [item["text"] for item in wikitext if item["text"].strip()]
    print(f"  WikiText-2: {len(wiki_texts)} examples")

    print("Downloading Alpaca dataset...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        alpaca_texts = []
        indices = list(range(len(alpaca)))
        random.seed(42)
        random.shuffle(indices)
        for i in indices[: config.alpaca_subset]:
            item = alpaca[i]
            prompt = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            if inp:
                prompt = f"{prompt} {inp}"
            alpaca_texts.append(f"{prompt} {output}")
        print(f"  Alpaca: {len(alpaca_texts)} examples (subsampled)")
    except Exception as e:
        print(f"  Alpaca download failed ({e}), skipping")
        alpaca_texts = []

    all_texts = ts_texts + wiki_texts + alpaca_texts
    print(f"Total raw texts: {len(all_texts)}")
    return all_texts


def step_tokenizer(all_texts, config):
    print("\n" + "=" * 60)
    print("STEP 1: Training Tokenizer")
    print("=" * 60)

    tokenizer_path = "outputs/tokenizer"
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print(f"Loaded existing tokenizer (vocab_size={tokenizer.vocab_size})")
        return tokenizer

    tokenizer = train_tokenizer(all_texts, vocab_size=config.tokenizer_vocab_size, save_path=tokenizer_path)
    return tokenizer


def step_datasets(tokenizer, all_texts, config):
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Datasets")
    print("=" * 60)

    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    print(f"Tokenizing {len(train_texts)} train texts...")
    train_dataset = TextDataset(tokenizer, train_texts, max_length=config.max_seq_length)
    print(f"Tokenizing {len(val_texts)} val texts...")
    val_dataset = TextDataset(tokenizer, val_texts, max_length=config.max_seq_length)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")
    return train_dataset, val_dataset


def step_model(model_config):
    print("\n" + "=" * 60)
    print("STEP 3: Creating Model")
    print("=" * 60)

    model = BurkiLLM(model_config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (fp32):    {total * 4 / 1024**2:.1f} MB")
    return model


def step_pretrain(model, train_dataset, val_dataset, config):
    print("\n" + "=" * 60)
    print("STEP 4: Pre-training")
    print("=" * 60)

    trainer = Trainer(model, train_dataset, val_dataset, config)
    best_loss = trainer.train(save_dir="outputs/checkpoints")
    print(f"\nBest pre-training val loss: {best_loss:.4f}")
    return model


def step_finetune(model, tokenizer, config):
    """Instruction fine-tune on identity + Q&A data.
    
    The fine-tuning format is:  <bos>INSTRUCTION<eos>RESPONSE<eos>
    This matches exactly what generate_text() feeds at inference time.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Instruction Fine-tuning")
    print("=" * 60)

    identity_data = InstructionDataset.load_identity_data("data/identity.json")

    # Repeat identity data more for better memorization of self-identity answers
    augmented = identity_data * 30
    random.shuffle(augmented)

    dataset = InstructionDataset(tokenizer, augmented, max_length=config.max_seq_length)
    split = int(len(dataset) * 0.9)
    train_subset = torch.utils.data.Subset(dataset, range(split))
    val_subset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    print(f"Instruction train examples: {len(train_subset)}")
    print(f"Instruction val examples:   {len(val_subset)}")

    ft_config = TrainingConfig(
        batch_size=32,
        learning_rate=3e-5,          # lower LR for fine-tuning to preserve pretrain knowledge
        weight_decay=0.01,
        num_epochs=20,               # more epochs since dataset is small
        num_warmup_steps=30,
        grad_clip=1.0,
        patience=8,                  # generous patience so it trains fully
        mixed_precision=config.mixed_precision,
        max_seq_length=config.max_seq_length,
        device=config.device,
    )

    trainer = Trainer(model, train_subset, val_subset, ft_config)
    trainer.train(save_dir="outputs/checkpoints")
    return model


def step_inference(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("STEP 6: Inference Tests")
    print("=" * 60)

    prompts = [
        "Hello!",
        "Who are you?",
        "What is AI?",
        "What is machine learning?",
        "What is Python?",
        "Tell me a joke.",
        "What is a transformer?",
        "What can you do?",
        "Introduce yourself.",
    ]

    os.makedirs("outputs", exist_ok=True)
    results = []

    for prompt in prompts:
        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9,
        )
        results.append(f"Prompt: {prompt}\nResponse: {response}\n")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

    with open("outputs/inference_results.txt", "w", encoding="utf-8") as f:
        f.write("BurkiLLM Inference Test Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(r + "\n")

    print(f"\nResults saved to outputs/inference_results.txt")


def main():
    print("=" * 60)
    print("BurkiLLM — Lightweight Transformer Language Model")
    print("by Muhammad Zeeshan Burki")
    print("=" * 60)

    device = get_device()
    training_config = TrainingConfig(device=device)

    # Step 1: Load data and train tokenizer
    all_texts = load_raw_texts(training_config)
    tokenizer = step_tokenizer(all_texts, training_config)

    # Update model config to match tokenizer
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"\nModel Config:")
    print(f"  vocab_size={model_config.vocab_size}, hidden={model_config.hidden_size}")
    print(f"  layers={model_config.num_hidden_layers}, heads={model_config.num_attention_heads}")
    print(f"  context={model_config.max_position_embeddings}, intermediate={model_config.intermediate_size}")

    # Step 2: Prepare datasets
    train_dataset, val_dataset = step_datasets(tokenizer, all_texts, training_config)

    # Step 3: Create model
    model = step_model(model_config)

    # Step 4: Pre-train
    model = step_pretrain(model, train_dataset, val_dataset, training_config)

    # Step 5: Fine-tune on instructions
    model = step_finetune(model, tokenizer, training_config)

    # Step 6: Run inference tests
    step_inference(model, tokenizer, device)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Checkpoints: outputs/checkpoints/")
    print("Tokenizer:   outputs/tokenizer/")
    print("Results:     outputs/inference_results.txt")
    print("\nTo chat with your model:")
    print("  python chat.py")


if __name__ == "__main__":
    main()
