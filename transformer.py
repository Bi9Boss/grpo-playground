"""A small decoder-only Transformer for learning and experimentation."""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention in which tokens can only see the past."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Compute query, key, and value together, then split them in forward().
        self.qkv_projection = nn.Linear(
            config.d_model, 3 * config.d_model, bias=False
        )
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        query, key, value = self.qkv_projection(x).chunk(3, dim=-1)

        # [batch, sequence, d_model] -> [batch, heads, sequence, head_dim]
        query = query.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        output = attention_weights @ value

        # Concatenate all attention heads back into d_model.
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        output = self.output_projection(output)
        return self.output_dropout(output)


class FeedForward(nn.Module):
    """The position-wise two-layer MLP used by each Transformer block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = 4 * config.d_model
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """A pre-norm attention block followed by a pre-norm feed-forward block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Transformer(nn.Module):
    """A decoder-only Transformer that maps token IDs to next-token logits."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, sequence, vocab_size]."""
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )

        position_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinary next-token cross-entropy loss."""
        logits = self(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from the model."""
        for _ in range(max_new_tokens):
            model_input = input_ids[:, -self.config.max_seq_len :]
            next_token_logits = self(model_input)[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids
