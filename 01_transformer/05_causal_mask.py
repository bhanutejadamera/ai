import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

embedding_size = 4

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

query_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False,
)

key_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False,
)

value_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False,
)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

scores = Q @ K.transpose(0, 1)

scaled_scores = scores / math.sqrt(
    embedding_size
)

print("Scores BEFORE masking:")
print(scaled_scores)

sequence_length = len(words)

mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length,
    ),
    diagonal=1,
).bool()

print()
print("Causal mask:")
print(mask)

masked_scores = scaled_scores.masked_fill(
    mask,
    float("-inf"),
)

print()
print("Scores AFTER masking:")
print(masked_scores)

attention_weights = F.softmax(
    masked_scores,
    dim=-1,
)

print()
print("Attention weights:")
print(attention_weights)

output = attention_weights @ V

print()
print("Attention output:")
print(output)

print()
print("Who can look at whom?")

for row_index, word in enumerate(words):

    print()
    print(f"{word} looks at:")

    for column_index, other_word in enumerate(words):

        weight = attention_weights[
            row_index,
            column_index
        ].item()

        print(
            f"  {other_word}: "
            f"{weight:.2%}"
        )