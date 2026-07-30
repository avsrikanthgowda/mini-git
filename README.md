# MiniGPT — Build Your Own Language Model from Scratch

A simple GPT-style language model built entirely from scratch using PyTorch. No external model downloads — trains and runs 100% locally on your laptop CPU.

## Architecture

- **Word-level tokenizer** with special tokens (`<start>`, `<end>`, `<sep>`, `<pad>`, `<unk>`)
- **Transformer decoder** — 2 layers, 4 attention heads, 128-dim embeddings (~600K params)
- **Trained on a curated Q&A corpus** of 130+ question-answer pairs

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
Training takes ~2-5 minutes on CPU. You'll see loss decreasing each epoch.

### 3. Chat with MiniGPT
```bash
python chat.py
```

## Project Structure
```
mini-gpt/
├── data/
│   ├── training_data.txt    # Q&A training corpus
│   └── vocab.py             # Word-level tokenizer
├── model/
│   ├── __init__.py
│   └── transformer.py       # MiniGPT Transformer model
├── train.py                 # Training script
├── chat.py                  # Interactive chat CLI
├── checkpoints/             # Saved model weights (after training)
├── requirements.txt
└── README.md
```

## Example
```
=================================
MiniGPT v1.0
=================================

You: hello

MiniGPT:
hello! how are you?

---------------------------------

You: what is python?

MiniGPT:
python is a programming language.

---------------------------------

You: quit
```
