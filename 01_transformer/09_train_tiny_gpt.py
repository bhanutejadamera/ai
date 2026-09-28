import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),
        )

    def forward(self, x):
        sequence_length = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
            ),
            diagonal=1,
        ).bool()

        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + attention_output

        normalized = self.norm2(x)

        ff_output = self.feed_forward(normalized)

        x = x + ff_output

        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=32,
        num_heads=4,
        num_layers=2,
        max_sequence_length=64,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size,
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_size)

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(position_ids)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.lm_head(x)

text = """
hello world
hello transformer
hello language model
transformers predict the next token
"""

characters = sorted(set(text))

char_to_id = {
    char: index
    for index, char in enumerate(characters)
}

id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

vocab_size = len(characters)

print("Vocabulary size:", vocab_size)
print("Vocabulary:", characters)

data = torch.tensor(
    [char_to_id[char] for char in text],
    dtype=torch.long,
)

sequence_length = 16


def get_batch(batch_size=8):
    starts = torch.randint(
        0,
        len(data) - sequence_length - 1,
        (batch_size,),
    )

    inputs = torch.stack([
        data[start:start + sequence_length]
        for start in starts
    ])

    targets = torch.stack([
        data[start + 1:start + sequence_length + 1]
        for start in starts
    ])

    return inputs, targets

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_size=32,
    num_heads=4,
    num_layers=2,
    max_sequence_length=64,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

for step in range(1000):
    inputs, targets = get_batch()

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f}"
        )

@torch.no_grad()
def generate(
    model,
    starting_text,
    max_new_characters=100,
):
    model.eval()

    token_ids = torch.tensor([
        [
            char_to_id[char]
            for char in starting_text
        ]
    ])

    for _ in range(max_new_characters):

        # Our model only supports sequences
        # up to 64 characters long.
        context = token_ids[:, -64:]

        logits = model(context)

        # We only care about the prediction
        # from the final position.
        next_token_logits = logits[
            0,
            -1,
            :
        ]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token_id = next_token_id.reshape(
            1,
            1,
        )

        token_ids = torch.cat(
            [
                token_ids,
                next_token_id,
            ],
            dim=1,
        )

    generated_text = "".join(
        id_to_char[token_id.item()]
        for token_id in token_ids[0]
    )

    return generated_text

print()
print("=== GENERATED TEXT ===")

result = generate(
    model=model,
    starting_text="hello",
    max_new_characters=100,
)

print(result)