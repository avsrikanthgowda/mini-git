import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data.vocab import Vocabulary, load_qa_pairs
from model.transformer import MiniGPTModel

# --- Config ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "training_data.txt")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "vocab.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "mini_gpt.pt")
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
MAX_SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 500
LR = 0.001
DEVICE = "cpu"


class QADataset(Dataset):
    def __init__(self, sequences, pad_idx, max_len):
        self.sequences = sequences
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate if too long
        seq = seq[:self.max_len]
        # Pad
        padded = seq + [self.pad_idx] * (self.max_len - len(seq))
        input_seq = padded[:-1]
        target_seq = padded[1:]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def build_sequences(qa_pairs, vocab):
    sequences = []
    for question, answer in qa_pairs:
        q_tokens = vocab.encode(question)
        a_tokens = vocab.encode(answer)
        # Format: <start> question <sep> answer <end>
        seq = [vocab.start_idx] + q_tokens + [vocab.sep_idx] + a_tokens + [vocab.end_idx]
        sequences.append(seq)

        # Data augmentation: add variant with question mark
        q_with_mark = question + " ?"
        q_tokens2 = vocab.encode(q_with_mark)
        seq2 = [vocab.start_idx] + q_tokens2 + [vocab.sep_idx] + a_tokens + [vocab.end_idx]
        sequences.append(seq2)

    return sequences


def train():
    print("=" * 40)
    print("  MiniGPT Training")
    print("=" * 40)

    # Load data
    print("\n[1/5] Loading training data...")
    qa_pairs = load_qa_pairs(DATA_PATH)
    print(f"  Loaded {len(qa_pairs)} Q&A pairs")

    # Build vocabulary
    print("[2/5] Building vocabulary...")
    vocab = Vocabulary()
    all_texts = []
    for q, a in qa_pairs:
        all_texts.append(q)
        all_texts.append(q + " ?")  # Include punctuation variant in vocab
        all_texts.append(a)
    vocab.build_vocab(all_texts)
    print(f"  Vocabulary size: {vocab.vocab_size}")

    # Save vocab
    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    vocab.save(VOCAB_PATH)

    # Build sequences
    print("[3/5] Preparing sequences...")
    sequences = build_sequences(qa_pairs, vocab)
    dataset = QADataset(sequences, vocab.pad_idx, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(dataset)} training sequences")

    # Create model
    print("[4/5] Creating MiniGPT model...")
    model = MiniGPTModel(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_SEQ_LEN,
        pad_idx=vocab.pad_idx,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Training
    print("[5/5] Training...\n")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        num_batches = 0

        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            logits = model(input_batch)
            loss = criterion(logits.reshape(-1, vocab.vocab_size), target_batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab.vocab_size,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "num_layers": NUM_LAYERS,
        "max_len": MAX_SEQ_LEN,
    }, MODEL_PATH)

    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Vocab saved to {VOCAB_PATH}")
    print("\n" + "=" * 40)
    print("  Training complete!")
    print("  Run 'python chat.py' to start chatting.")
    print("=" * 40)


if __name__ == "__main__":
    train()
