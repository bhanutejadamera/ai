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


prompt = "Explain what a Transformer is in one sentence."


messages = [
    {
        "role": "user",
        "content": prompt,
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)


inputs = tokenizer(
    text,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


with torch.no_grad():

    generated = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )


new_tokens = generated[
    0,
    inputs["input_ids"].shape[1]:
]


response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)


print("User:")
print(prompt)

print()

print("Model:")
print(response)