import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


print("Device:", device)


print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()


print()
print("Model loaded successfully.")