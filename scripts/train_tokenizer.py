"""Standalone tokenizer training script."""

import sys
from datasets import load_dataset
from data.tokenizer import train_tokenizer


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    save_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/tokenizer"

    print("Loading training texts...")
    ts = load_dataset("roneneldan/TinyStories", split="train[:50000]", trust_remote_code=True)
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)

    texts = [item["text"] for item in ts if item["text"].strip()]
    texts += [item["text"] for item in wiki if item["text"].strip()]
    print(f"Training tokenizer on {len(texts)} texts (vocab_size={vocab_size})")

    tokenizer = train_tokenizer(texts, vocab_size=vocab_size, save_path=save_path)
    print(f"Done. Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
