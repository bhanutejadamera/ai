import torch
import torch.nn as nn


token_ids = torch.tensor([
    0,  # I
    1,  # like
    2,  # bananas
])


vocab_size = 3
embedding_size = 4


token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_size,
)


token_vectors = token_embedding(token_ids)


print("Token embeddings:")
print(token_vectors)

print()
print("Shape:")
print(token_vectors.shape)

position_ids = torch.tensor([
    0,
    1,
    2,
])

print()
print("Position IDs:")
print(position_ids)

max_positions = 10


position_embedding = nn.Embedding(
    num_embeddings=max_positions,
    embedding_dim=embedding_size,
)


position_vectors = position_embedding(position_ids)


print()
print("Position embeddings:")
print(position_vectors)

combined = token_vectors + position_vectors


print()
print("Token + position embeddings:")
print(combined)

print()
print("Shape:")
print(combined.shape)

i_token_id = torch.tensor(0)

first_position = torch.tensor(0)
third_position = torch.tensor(2)


i_as_first_word = (
    token_embedding(i_token_id)
    + position_embedding(first_position)
)

i_as_third_word = (
    token_embedding(i_token_id)
    + position_embedding(third_position)
)


print()
print("'I' at position 0:")
print(i_as_first_word)

print()
print("'I' at position 2:")
print(i_as_third_word)