import json
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Chunks raw text into fixed-length sequences for causal LM pre-training."""

    def __init__(self, tokenizer, texts, max_length=256):
        self.examples = []
        all_tokens = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)

        # Chunk into sequences of max_length + 1 (input + 1 shifted target)
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = all_tokens[i : i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels    = torch.tensor(tokens[1:],  dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class InstructionDataset(Dataset):
    """Formats instruction/response pairs for fine-tuning.

    Format:  <bos>INSTRUCTION<eos>RESPONSE<eos>
    Labels:  -100 for the instruction tokens (we only train on the response)
    """

    def __init__(self, tokenizer, examples, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        for ex in examples:
            instruction = ex["instruction"]
            response    = ex["response"]

            # Encode the instruction prefix:  <bos>INSTRUCTION<eos>
            prefix_text = f"{tokenizer.bos_token}{instruction}{tokenizer.eos_token}"
            prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)

            # Encode the response suffix:  RESPONSE<eos>
            suffix_text = f"{response}{tokenizer.eos_token}"
            suffix_ids  = tokenizer.encode(suffix_text, add_special_tokens=False)

            all_ids = prefix_ids + suffix_ids

            if len(all_ids) < 4:
                continue

            # Truncate if needed
            if len(all_ids) > max_length + 1:
                all_ids = all_ids[: max_length + 1]

            # Build labels: mask out the instruction part with -100
            # so loss is computed only on the response tokens
            labels = [-100] * len(prefix_ids) + suffix_ids
            if len(labels) > max_length + 1:
                labels = labels[: max_length + 1]

            # Pad / align lengths (both should be same already)
            self.examples.append({
                "input_ids": all_ids[:-1],
                "labels":    labels[1:],   # shifted by 1 for next-token prediction
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels    = torch.tensor(ex["labels"],    dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def load_identity_data(path="data/identity.json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def collate_fn(batch, pad_token_id=0):
    input_ids = [item["input_ids"] for item in batch]
    labels    = [item["labels"]    for item in batch]
    max_len   = max(len(ids) for ids in input_ids)

    padded_inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100,          dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len),               dtype=torch.long)

    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        padded_inputs[i,  : len(ids)]  = ids
        padded_labels[i,  : len(lbls)] = lbls
        attention_mask[i, : len(ids)]  = 1

    return {
        "input_ids":      padded_inputs,
        "labels":         padded_labels,
        "attention_mask": attention_mask,
    }
