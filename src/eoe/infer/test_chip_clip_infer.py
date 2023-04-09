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
    # device = "cpu"
    images = []
    if pt.isdir(img_path):
        images = [pt.join(img_path, f) for f in os.listdir(img_path)]
    else:
        images.append(img_path)
    # read labels
    label_file_path = os.path.join(img_path, '..', 'labels')
    glogger.info("label_file_path {}", label_file_path)
    lable_dict = {}
    if os.path.exists(label_file_path):
        label_files = [pt.join(label_file_path, f) for f in os.listdir(label_file_path)]
        for label_file in label_files:
            with open(label_file, 'r') as f:
                key = Path(label_file).stem
                label = f.readline().rstrip()
                lable_dict[key] = label
    glogger.info("lable_dict {}", lable_dict)
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
        
        if img_name in lable_dict:
            raw_label_id = int(lable_dict[img_name])
        # if jit_probs.tolist()[1] > .62:
        #     best_idx = 1
        # else:
        #     best_idx = 0
        # glogger.info(jit_probs.tolist())
        probs = {"good": jit_probs.tolist()[0], "bad": jit_probs.tolist()[1]}
        if raw_label_id != best_idx:
            glogger.info("bad infer image {}, predict probs {} predict label {} raw_label {}", img_name, probs, labels[best_idx], raw_label_id)
        # glogger.info("image {}, predict probs {} predict label {}", img_name, probs, labels[best_idx])

if __name__ == '__main__':
    img_path = "/opt/eoe/data/datasets/chip/images/GOOD3.jpg"
    img_path = "/opt/eoe/data/datasets/chip/images/NG3.jpg"
    img_path = "/opt/eoe/data/datasets/chip/images"
    # img_path = "/opt/eoe/data/datasets/chip_03_26/images"
    # img_path = "/opt/eoe/data/datasets/chip_old/images"
        # img_path = "/opt/eoe/data/datasets/chip_old/images"
    model_file = "/tmp/good.pt"
    model_file = "/tmp/good_03_26.pt"
    model_file = "/opt/eoe/data/models/03_26_good.pt"
    # model_file = "/opt/eoe/data/models/03_26_bad.pt"
    
    model_file = "/opt/eoe/data/models/panzi_good.pt"
    model_file = "/opt/eoe/data/models/16c_0405_1_good.pt"
    img_path = "/opt/eoe/data/datasets/chip/test/images"
    
    model_file = "/opt/eoe/data/models/0409_good.pt"
    # model_file = "/opt/eoe/data/models/0409_good_2.pt"
    img_path = "/opt/eoe/data/datasets/chip/test/images"
    infer(img_path, model_file)