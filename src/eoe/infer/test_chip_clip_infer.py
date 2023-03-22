import numpy as np
import torch
from PIL import Image
import os
import eoe.models.clip_official.clip as official_clip
from loguru import logger as glogger
import os.path as pt
from pathlib import Path


def load_model(model_file, device):
    model, transform = official_clip.load(model_file, device, jit=True)
    return model, transform


def infer(img_path, model_file):
    device = "cuda"
    images = []
    if pt.isdir(img_path):
        images = [pt.join(img_path, f) for f in os.listdir(img_path)]
        ...
    else:
        images.append(img_path)
    glogger.info("images {}", images)
    # return
    # for each
    model, transform = load_model(model_file, device)
    
    labels = ["good", "bad"]
   
    for img in images:
        img_name = Path(img).stem
        image = transform(Image.open(img)).unsqueeze(0).to(device)
        text = official_clip.tokenize(["a photo of a good", "a photo of a something"]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        best_idx = np.argmax(jit_probs)
        # glogger.info(jit_probs.tolist())
        probs = {"good": jit_probs.tolist()[0], "bad": jit_probs.tolist()[1]}
        glogger.info("image {}, predict probs {} predict label {}", img_name, probs, labels[best_idx])

if __name__ == '__main__':
    img_path = "/opt/eoe/data/datasets/chip/images/GOOD3.jpg"
    img_path = "/opt/eoe/data/datasets/chip/images/NG3.jpg"
    img_path = "/opt/eoe/data/datasets/chip/images"
    model_file = "/tmp/good.pt"
    infer(img_path, model_file)