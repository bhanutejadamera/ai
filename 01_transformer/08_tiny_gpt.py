import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_size,
                embedding_size * 4,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_size * 4,
                embedding_size,
            ),
        )

    def forward(self, x):
        sequence_length = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
            ),
            diagonal=1,
        ).bool()

        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + attention_output

        normalized = self.norm2(x)

        ff_output = self.feed_forward(
            normalized
        )

        x = x + ff_output

        return x

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=32,
        num_heads=4,
        num_layers=2,
        max_sequence_length=64,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size,
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_size
        )

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
        )

    def forward(self, token_ids):

        batch_size, sequence_length = (
            token_ids.shape
        )

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            position_ids
        )

        x = (
            token_vectors
            + position_vectors
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


torch.manual_seed(42)

vocab_size = 10

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_size=32,
    num_heads=4,
    num_layers=2,
)

token_ids = torch.tensor([
    [0, 1, 2]
])

logits = model(token_ids)

print("Input shape:")
print(token_ids.shape)

print()

print("Output shape:")
print(logits.shape)

last_token_logits = logits[
    0,
    -1,
    :
]

print()
print("Scores for next token:")
print(last_token_logits)

probabilities = torch.softmax(
    last_token_logits,
    dim=-1,
)

print()
print("Probabilities:")
print(probabilities)

print()
print("Probability total:")
print(probabilities.sum())

predicted_token = torch.argmax(
    probabilities
)

print()
print("Predicted token ID:")
print(predicted_token.item())