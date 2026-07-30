import os
import sys
import re
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data.vocab import Vocabulary
from model.transformer import MiniGPTModel

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "vocab.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "mini_gpt.pt")
DEVICE = "cpu"


def load_model():
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    model = MiniGPTModel(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        ff_dim=checkpoint["ff_dim"],
        num_layers=checkpoint["num_layers"],
        max_len=checkpoint["max_len"],
        pad_idx=vocab.pad_idx,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def normalize_input(text):
    """Normalize user input to match training data format."""
    text = text.lower().strip()
    # Remove punctuation at end
    text = re.sub(r'[?.!,;:]+$', '', text)
    # Remove common prefixes that aren't part of the question
    prefixes = ['hey ', 'hi ', 'hello ', 'please ', 'can you tell me ',
                'tell me ', 'could you ', 'can you ', 'do you know ']
    for prefix in prefixes:
        if text.startswith(prefix) and len(text) > len(prefix) + 5:
            stripped = text[len(prefix):]
            # Only strip if the remaining part looks like a question
            if any(stripped.startswith(w) for w in ['what', 'who', 'when', 'where', 'how', 'why', 'which']):
                text = stripped
                break
    # Remove internal punctuation
    text = re.sub(r'[?.!,;:]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_response(model, vocab, user_input, temperature=0.3, max_tokens=50):
    user_input = normalize_input(user_input)
    q_tokens = vocab.encode(user_input)
    input_ids = [vocab.start_idx] + q_tokens + [vocab.sep_idx]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    output = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        end_idx=vocab.end_idx,
        temperature=temperature,
    )

    all_ids = output[0].tolist()

    # Find the <sep> token and extract only the answer part
    try:
        sep_pos = all_ids.index(vocab.sep_idx)
        answer_ids = all_ids[sep_pos + 1:]
    except ValueError:
        answer_ids = all_ids

    # Remove special tokens
    answer_ids = [i for i in answer_ids if i not in (vocab.start_idx, vocab.end_idx, vocab.pad_idx, vocab.sep_idx)]

    response = vocab.decode(answer_ids)
    return response


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("Error: Model not found. Please run 'python train.py' first.")
        sys.exit(1)

    print("\nLoading MiniGPT...")
    model, vocab = load_model()

    print()
    print("=" * 33)
    print("MiniGPT v1.0")
    print("=" * 33)
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! Have a great day!")
            break

        response = generate_response(model, vocab, user_input)
        print(f"\nMiniGPT:\n{response}\n")
        print("-" * 33)
        print()


if __name__ == "__main__":
    main()
