import torch
from transformers import PreTrainedTokenizerFast
from configs.model_config import ModelConfig
from models.model import BurkiLLM
from inference.generate import load_model_for_inference, generate_text

CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
TOKENIZER_PATH  = "outputs/tokenizer"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
print(f"Loading model from {CHECKPOINT_PATH} on {device}...")
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model  = load_model_for_inference(CHECKPOINT_PATH, config, device)
print("Ready!  Type your message below (type 'exit' to quit).\n")

while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not prompt:
        continue
    response = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.9,
    )
    print(f"BurkiLLM: {response}\n")
