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
    prefixes = ['can you tell me ', 'do you know ', 'could you ',
                'can you ', 'tell me ', 'please ', 'hello ', 'hey ', 'hi ']
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


def clean_response(text):
    """Remove repeated phrases/words from model output."""
    words = text.split()
    if len(words) < 4:
        return text
    # Detect repeating n-gram loops (2-6 word patterns)
    for n in range(2, 7):
        if len(words) >= n * 2:
            for i in range(len(words) - n):
                pattern = words[i:i + n]
                # Check if pattern repeats immediately after
                j = i + n
                repeats = 0
                while j + n <= len(words) and words[j:j + n] == pattern:
                    repeats += 1
                    j += n
                if repeats >= 1:
                    words = words[:i + n]
                    return ' '.join(words)
    return text


def generate_response(model, vocab, user_input, temperature=0.5, max_tokens=50,
                      top_k=20, repetition_penalty=1.3, num_samples=3):
    user_input = normalize_input(user_input)
    q_tokens = vocab.encode(user_input)
    input_ids = [vocab.start_idx] + q_tokens + [vocab.sep_idx]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    candidates = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                end_idx=vocab.end_idx,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            all_ids = output[0].tolist()

            try:
                sep_pos = all_ids.index(vocab.sep_idx)
                answer_ids = all_ids[sep_pos + 1:]
            except ValueError:
                answer_ids = all_ids

            answer_ids = [i for i in answer_ids if i not in
                          (vocab.start_idx, vocab.end_idx, vocab.pad_idx, vocab.sep_idx)]

            response = clean_response(vocab.decode(answer_ids).strip())
            if response:
                candidates.append(response)

    if not candidates:
        return "I'm not sure how to answer that."

    # Pick the longest non-degenerate response
    best = max(candidates, key=lambda r: len(r.split()))
    return best[0].upper() + best[1:]


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("Error: Model not found. Please run 'python train.py' first.")
        sys.exit(1)

    print("\nLoading MiniGPT...")
    model, vocab = load_model()

    print()
    print("=" * 50)
    print("  MiniGPT v1.0 — Airspan Networks")
    print("=" * 50)
    print()
    print("  Trained on 349 Q&A pairs covering:")
    print("  - Programming   : Python, Java, JS, C/C++,")
    print("                    Rust, Go, Ruby, Swift & more")
    print("  - Web Dev       : HTML, CSS, React, Angular,")
    print("                    Django, Flask, Node.js")
    print("  - AI / ML       : Neural Networks, Deep Learning,")
    print("                    NLP, PyTorch, TensorFlow")
    print("  - Databases     : SQL, MongoDB, PostgreSQL, Redis")
    print("  - DevOps / Cloud: Docker, Kubernetes, AWS, Azure,")
    print("                    CI/CD, Git, Linux")
    print("  - CS Fundamentals: Data Structures, Algorithms,")
    print("                    OOP, Testing, Networking")
    print()
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)
    print()

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
