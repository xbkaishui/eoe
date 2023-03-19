import numpy as np
import torch
from PIL import Image
import os
import eoe.models.clip_official.clip as official_clip
from loguru import logger

def test_clip_infer(model_name):
    cur_file = os.path.dirname(__file__)
    device = "cpu"
    jit_model, transform = official_clip.load(model_name, device=device, jit=False)

    image = transform(Image.open(f"{cur_file}/CLIP.png")).unsqueeze(0).to(device)
    text = official_clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = jit_model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    logger.info(jit_probs.tolist())

if __name__ == '__main__':
    models = official_clip.available_models()
    logger.info("support models {}", models)
    test_clip_infer("ViT-B/32")