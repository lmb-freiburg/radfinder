"""Minimal smoke test for the radfinder VL model.

Load the exported HF model and run a dummy forward pass.
"""

import torch
from transformers import AutoModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("lmb-freiburg/radfinder", trust_remote_code=True)
    model.eval()
    model.to(device)

    # Dummy CT input: (batch_size * crops, channels, height, width, depth).
    # For a (3 x 3 x 4) grid of (128 x 128 x 32) patches → scan (384 x 384 x 128).
    pixel_values = torch.randn(36, 1, 128, 128, 32, device=device)

    # Dummy tokenized text (use a real Qwen3 tokenizer for actual inference).
    input_ids = torch.randint(0, 1000, (1, 16), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(pixel_values, input_ids, attention_mask, grid_size=((3, 3, 4),))

    print("image_embeddings:", out["image_embeddings"].shape)
    print("text_embeddings:", out["text_embeddings"].shape)


if __name__ == "__main__":
    main()
