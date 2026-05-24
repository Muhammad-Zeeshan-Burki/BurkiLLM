import torch
from models.model import BurkiLLM


def load_model_for_inference(model_path, config, device="cpu"):
    model = BurkiLLM(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9):
    """Generate a response to a prompt using instruction format.

    Format matches fine-tuning:  <bos>INSTRUCTION<eos>RESPONSE<eos>

    NOTE: We decode the FULL token sequence (prompt + response) together, then
    strip the instruction prefix as a string.  Decoding only the tail slice of
    a ByteLevel-BPE sequence corrupts the leading bytes of every generated word.
    """
    device = next(model.parameters()).device

    bos = tokenizer.bos_token  # "<bos>"
    eos = tokenizer.eos_token  # "<eos>"

    # Instruction prefix exactly as used during fine-tuning
    formatted = f"{bos}{prompt}{eos}"
    encoded   = tokenizer.encode(formatted, add_special_tokens=False)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ── Decode the FULL sequence with special tokens visible ──────────────
    full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)

    # Strip leading BOS
    if full_text.startswith(bos):
        full_text = full_text[len(bos):]

    # Find the EOS that closes the instruction and take everything after it
    first_eos = full_text.find(eos)
    if first_eos != -1:
        response = full_text[first_eos + len(eos):]
    else:
        response = full_text  # fallback: return everything

    # Clean up any trailing special tokens and whitespace
    response = response.replace(eos, "").replace(bos, "").strip()
    return response
