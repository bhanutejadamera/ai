import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


# Use your M4 GPU if available.
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("Using device:", device)
print("Loading model...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
).to(device)


model.eval()


# This Python list is our conversation history.
messages = [
    {
        "role": "system",
        "content": (
            "You are a concise and friendly "
            "AI programming teacher."
        ),
    }
]


print()
print("Local chatbot ready.")
print("Type 'exit' to stop.")


while True:

    print()

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add the user's new message to history.
    messages.append({
        "role": "user",
        "content": user_input,
    })


    # Turn the conversation into the exact
    # token format Qwen expects.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )


    # Move tensors onto the M4 GPU.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )


    # generated contains:
    #
    # original conversation tokens
    # +
    # new assistant tokens
    #
    # We only want the new part.
    input_length = inputs[
        "input_ids"
    ].shape[1]


    new_tokens = generated[
        0,
        input_length:
    ]


    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )


    print()
    print("Assistant:", response)


    # THIS is what makes the next turn
    # remember the assistant's response.
    messages.append({
        "role": "assistant",
        "content": response,
    })