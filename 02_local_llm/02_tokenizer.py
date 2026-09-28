from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


text = "Transformers are amazing!"


token_ids = tokenizer.encode(
    text,
    add_special_tokens=False,
)


tokens = tokenizer.convert_ids_to_tokens(
    token_ids
)


print("Original text:")
print(text)

print()

print("Token IDs:")
print(token_ids)

print()

print("Tokens:")
print(tokens)

print()

print("Decoded again:")
print(
    tokenizer.decode(token_ids)
)