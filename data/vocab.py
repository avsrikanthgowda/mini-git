import re
import json
import os


class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    SEP_TOKEN = "<sep>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN, self.SEP_TOKEN]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @staticmethod
    def tokenize(text):
        text = text.lower().strip()
        text = re.sub(r"([?.!,;:'\"\-\(\)])", r" \1 ", text)
        tokens = text.split()
        return tokens

    def build_vocab(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1

        for word in sorted(self.word_freq.keys()):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        tokens = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word in (self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN):
                continue
            tokens.append(word)
        text = " ".join(tokens)
        # Clean up spacing around punctuation
        text = re.sub(r"\s+([?.!,;:'\"\-\)])", r"\1", text)
        text = re.sub(r"([\(\"])\s+", r"\1", text)
        return text

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @property
    def pad_idx(self):
        return self.word2idx[self.PAD_TOKEN]

    @property
    def start_idx(self):
        return self.word2idx[self.START_TOKEN]

    @property
    def end_idx(self):
        return self.word2idx[self.END_TOKEN]

    @property
    def sep_idx(self):
        return self.word2idx[self.SEP_TOKEN]

    def save(self, path):
        data = {"word2idx": self.word2idx, "idx2word": {int(k): v for k, v in self.idx2word.items()}}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}


def load_qa_pairs(filepath):
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        q_line = None
        a_line = None
        for line in lines:
            if line.startswith("Q:"):
                q_line = line[2:].strip()
            elif line.startswith("A:"):
                a_line = line[2:].strip()
        if q_line and a_line:
            pairs.append((q_line, a_line))
    return pairs
