import torch
import torch.nn as nn


torch.manual_seed(42)

words = [
    "I",
    "like",
    "bananas",
]

token_ids = torch.tensor([
    0,
    1,
    2,
])

embedding_size = 8

token_embedding = nn.Embedding(
    num_embeddings=3,
    embedding_dim=embedding_size,
)

position_embedding = nn.Embedding(
    num_embeddings=10,
    embedding_dim=embedding_size,
)

position_ids = torch.tensor([
    0,
    1,
    2,
])

x = (
    token_embedding(token_ids)
    + position_embedding(position_ids)
)

print("Input shape:")
print(x.shape)

attention = nn.MultiheadAttention(
    embed_dim=embedding_size,
    num_heads=2,
    batch_first=True,
)

x = x.unsqueeze(0)

print()
print("After adding batch dimension:")
print(x.shape)

sequence_length = x.size(1)

causal_mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length,
    ),
    diagonal=1,
).bool()

output, attention_weights = attention(
    x,
    x,
    x,
    attn_mask=causal_mask,
    need_weights=True,
)

print()
print("Output shape:")
print(output.shape)

print()
print("Attention weights:")
print(attention_weights)