import torch
import torch.nn as nn


token_ids = torch.tensor([
    0,  # I
    1,  # like
    2,  # bananas
])

print("Token IDs:")
print(token_ids)

vocab_size = 3
embedding_size = 4

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_size,
)

print()
print("Embedding table:")
print(embedding.weight)

vectors = embedding(token_ids)

print()
print("Token embeddings:")
print(vectors)

print()
print("Shape:")
print(vectors.shape)

words = {
    0: "I",
    1: "like",
    2: "bananas",
}

print()
print("Each token and its vector:")

for token_id in token_ids:
    token_number = token_id.item()
    word = words[token_number]
    vector = embedding(token_id)

    print()
    print(word)
    print(vector)