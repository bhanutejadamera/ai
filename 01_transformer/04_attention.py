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


vocab_size = 3
embedding_size = 4


token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
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


token_vectors = token_embedding(token_ids)

position_vectors = position_embedding(position_ids)

x = token_vectors + position_vectors


print("Input vectors:")
print(x)

print()
print("Shape:")
print(x.shape)

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


print()
print("Queries:")
print(Q)

print()
print("Keys:")
print(K)

print()
print("Values:")
print(V)

scores = Q @ K.transpose(0, 1)


print()
print("Raw attention scores:")
print(scores)

print()
print("Score shape:")
print(scores.shape)

scaled_scores = scores / math.sqrt(embedding_size)


print()
print("Scaled attention scores:")
print(scaled_scores)

attention_weights = F.softmax(
    scaled_scores,
    dim=-1,
)


print()
print("Attention weights:")
print(attention_weights)

print()
print("Each row sums to:")
print(attention_weights.sum(dim=-1))

print()
print("Who is paying attention to whom?")

for row_index, word in enumerate(words):

    print()
    print(f"{word} is looking at:")

    for column_index, other_word in enumerate(words):

        weight = attention_weights[
            row_index,
            column_index
        ].item()

        print(
            f"  {other_word}: "
            f"{weight:.2%}"
        )

output = attention_weights @ V


print()
print("Attention output:")
print(output)

print()
print("Output shape:")
print(output.shape)