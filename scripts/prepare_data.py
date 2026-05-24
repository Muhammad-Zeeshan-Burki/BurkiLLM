"""Standalone data preparation — downloads and caches datasets."""

import os
import sys
import random
import json
from datasets import load_dataset


def download_and_prepare(output_dir="data/prepared", tinystories_n=100_000, alpaca_n=5_000):
    os.makedirs(output_dir, exist_ok=True)

    # TinyStories
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    texts = [item["text"] for item in ds if item["text"].strip()]
    if len(texts) > tinystories_n:
        random.seed(42)
        texts = random.sample(texts, tinystories_n)
    print(f"  Kept {len(texts)} TinyStories examples")

    with open(os.path.join(output_dir, "tinystories.txt"), "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n")

    # WikiText-2
    print("Downloading WikiText-2...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    wiki_texts = [item["text"] for item in wiki if item["text"].strip()]
    print(f"  Kept {len(wiki_texts)} WikiText-2 lines")

    with open(os.path.join(output_dir, "wikitext2.txt"), "w", encoding="utf-8") as f:
        for t in wiki_texts:
            f.write(t.strip() + "\n")

    # Alpaca
    print("Downloading Alpaca...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        indices = list(range(len(alpaca)))
        random.seed(42)
        random.shuffle(indices)
        alpaca_data = []
        for i in indices[:alpaca_n]:
            item = alpaca[i]
            alpaca_data.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            })
        with open(os.path.join(output_dir, "alpaca_subset.json"), "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, indent=2)
        print(f"  Kept {len(alpaca_data)} Alpaca examples")
    except Exception as e:
        print(f"  Alpaca download failed: {e}")

    print(f"\nData saved to {output_dir}/")


if __name__ == "__main__":
    download_and_prepare()
