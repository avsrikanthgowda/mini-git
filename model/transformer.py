import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256,
                 num_layers=2, max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        pad_mask = (x == self.pad_idx)

        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)

        # Use decoder with self-attention only (memory = same as target)
        out = self.transformer_decoder(
            tgt=x_emb,
            memory=x_emb,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )

        logits = self.output_proj(out)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, end_idx=3, temperature=0.7):
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.size(1) >= self.max_len:
                break

            logits = self.forward(generated)
            next_logits = logits[:, -1, :]

            if temperature <= 0.1:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == end_idx:
                break

        return generated
