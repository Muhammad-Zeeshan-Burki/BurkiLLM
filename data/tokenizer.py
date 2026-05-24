import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_tokenizer(texts, vocab_size=8192, save_path="outputs/tokenizer"):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer, length=len(texts))

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )

    os.makedirs(save_path, exist_ok=True)
    wrapped.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={wrapped.vocab_size})")
    return wrapped
