import numpy as np
import torch
from PIL import Image
import os
import eoe.models.clip_official.clip as official_clip
from loguru import logger as glogger
import os.path as pt
from pathlib import Path
import cv2 as cv


def load_model(model_file, device):
    model, transform = official_clip.load(model_file, device, jit=True)
    return model, transform


def infer(img_path, model_file, output_path):
    # device = "cuda"
    device = "cpu"
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
    
    result_texts = ["GOOD", "NG"]
   
    for img in images:
        img_name = Path(img).stem
        output_file_path = f'{output_path}/{img_name}.jpg'
        image = transform(Image.open(img)).unsqueeze(0).to(device)
        text = official_clip.tokenize(["a photo of a good", "a photo of a something"]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        # best_idx = np.argmax(jit_probs)
        if jit_probs.tolist()[1] > .62:
            best_idx = 1
        else:
            best_idx = 0
        # glogger.info(jit_probs.tolist())
        probs = {"good": jit_probs.tolist()[0], "bad": jit_probs.tolist()[1]}
        glogger.info("image {}, predict probs {} predict label {}", img_name, probs, labels[best_idx])
        # write output file
        src = cv.imread(img)
        result_text = result_texts[best_idx]
        AddText = src.copy()
        size = (100,100)
        scale = 1.5
        w, h, _ = src.shape
        weight = 5
        if w > 1000:
            scale = 5
            size = (300, 300)
        elif w > 3000:
            scale = 20
            size = (500, 500)
            weight = 30
        cv.putText(AddText, result_text, size, cv.FONT_HERSHEY_COMPLEX, scale, (0, 0, 255), weight)
        # cv.putText(AddText, result_text, (200, 200), cv.FONT_HERSHEY_COMPLEX, 4.0, (0, 0, 255), 5)
        cv.imwrite(output_file_path, AddText)

if __name__ == '__main__':
    img_path = "/opt/eoe/data/datasets/chip/images"
    model_file = "/opt/eoe/data/models/panzi_good.pt"
    output_path = "/tmp/images/panzi";
    infer(img_path, model_file, output_path)