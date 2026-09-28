import torch
import torch.nn as nn


torch.manual_seed(42)


# Imagine these are already the output
# from our attention layer.
x = torch.randn(
    1,   # one sentence
    3,   # three tokens
    8,   # eight numbers per token
)


print("Input:")
print(x)

print()
print("Input shape:")
print(x.shape)

feed_forward = nn.Sequential(
    nn.Linear(8, 32),
    nn.GELU(),
    nn.Linear(32, 8),
)

ff_output = feed_forward(x)

print()
print("Feed-forward output:")
print(ff_output)

print()
print("Output shape:")
print(ff_output.shape)

residual_output = x + ff_output

print()
print("After residual connection:")
print(residual_output)

layer_norm = nn.LayerNorm(8)

normalized = layer_norm(
    residual_output
)

print()
print("After LayerNorm:")
print(normalized)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size=8,
        num_heads=2,
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
        # Self-attention with residual connection and LayerNorm
        attention_output, _ = self.attention(
            x,
            x,
            x,
        )
        x = self.norm1(x + attention_output)

        # Feed-forward with residual connection and LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x
    
block = TransformerBlock(
    embedding_size=8,
    num_heads=2,
)

result = block(x)

print()
print("Transformer block output:")
print(result)

print()
print("Transformer block shape:")
print(result.shape)