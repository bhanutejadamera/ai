import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
).to(device)


model.eval()


text = "The capital of France is"


inputs = tokenizer(
    text,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


with torch.no_grad():
    output = model(**inputs)


print("Input IDs shape:")
print(inputs["input_ids"].shape)


print()

print("Logits shape:")
print(output.logits.shape)

last_logits = output.logits[
    0,
    -1,
    :
]


probabilities = torch.softmax(
    last_logits,
    dim=-1,
)


top_probabilities, top_token_ids = torch.topk(
    probabilities,
    k=10,
)


print()
print("Top next-token predictions:")


for probability, token_id in zip(
    top_probabilities,
    top_token_ids,
):
    token_text = tokenizer.decode(
        [token_id.item()]
    )

    print(
        repr(token_text),
        f"{probability.item():.2%}"
    )