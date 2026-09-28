sentence = "I like bananas"

print("Original sentence:")
print(sentence)

# Split the sentence into tokens based on whitespace
tokens = sentence.split()

print()
print("Tokens:")
print(tokens)

# lookup table for mapping tokens to unique IDs
vocabulary = {
    "I": 0,
    "like": 1,
    "bananas": 2,
}

# Convert tokens to their corresponding IDs using the vocabulary
token_ids = [
    vocabulary[token]
    for token in tokens
]

print()
print("Vocabulary:")
print(vocabulary)

print()
print("Token IDs:")
print(token_ids)

# Reverse mapping from token IDs to tokens
id_to_token = {
    0: "I",
    1: "like",
    2: "bananas",
}

# Decode the token IDs back to their corresponding tokens using the reverse mapping
decoded_tokens = [
    id_to_token[token_id]
    for token_id in token_ids
]

decoded_sentence = " ".join(decoded_tokens)

print()
print("Decoded:")
print(decoded_sentence)