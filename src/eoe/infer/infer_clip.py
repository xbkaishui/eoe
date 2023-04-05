import numpy as np
import torch
from PIL import Image
import os
import eoe.models.clip_official.clip as official_clip
from loguru import logger as glogger


def infer(img_path, model_file):
    device = "cuda"
    model, transform = official_clip.load(model_file, device, jit=True)
    image = transform(Image.open(img_path)).unsqueeze(0).to(device)
    glogger.info("shape {}", image.shape)
    text = official_clip.tokenize(["a photo of a good", "a photo of a something"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    glogger.info(jit_probs.tolist())

if __name__ == '__main__':
    img_path = "/opt/eoe/data/datasets/chip/images/GOOD3.jpg"
    img_path = "/opt/eoe/data/datasets/chip/images/NG3.jpg"
    model_file = "/tmp/good.pt"
    # good
    img_path = "/opt/eoe/data/datasets/chip/images/0004-B-3.jpg"
    # bad
    # img_path = "/opt/eoe/data/datasets/chip/images/0034-T-3.jpg"
    model_file = "/opt/eoe/data/models/03_26_good.pt"
    model_file = "/opt/eoe/data/models/f4_bad.pt"
    # model_file = "/opt/eoe/data/models/f4_good.pt"
    img_path = "/opt/eoe/data/datasets/chip/images/bottom-0004-B-7.jpg"
    # img_path = "/opt/eoe/data/datasets/chip/images/NG-bottom-0004-B-6.jpg"
    infer(img_path, model_file)