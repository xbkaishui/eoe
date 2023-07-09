import os, sys
from addict import Dict
import torch, json
import numpy as np

from typing import Union
from pathlib import Path

from PIL import Image
import time
from loguru import logger
import eoe.models.clip_official.clip as official_clip

class TorchSimpleInferencer(object):
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        """ model checkpoint is self contained, so we can load it directly """
        self._checkpoint_path = checkpoint_path
        self.model, self.transform = self.load_model()


    def load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self._checkpoint_path, map_location='cpu')
        model, transform = official_clip.load_state_dict(ckpt['net'], self.device, jit=False)
        return model, transform
    
    def infer_img(self, img_path, threshold=0.3):
        # load label id map 
        if type(img_path) == str:
            # read image from path
            image = Image.open(img_path).convert("RGB")
            logger.info("image shape {}", image.size)
        else:
            image = img_path
        # transform images
        image = self.transform(image).unsqueeze(0).to(self.device)
        logger.info("shape {}", image.shape)
        text = official_clip.tokenize(["a photo of a good", "a photo of a something"]).to(self.device)

        labels = ["good", "bad"]
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).squeeze().cpu().numpy()
        probs_list = jit_probs.tolist()
        logger.info("probs {}", probs_list)
        best_idx = np.argmax(jit_probs)
        logger.info("best idx {}", best_idx)
        if probs_list[best_idx] < threshold:
            best_idx = 1 - best_idx
        probs = {"good": probs_list[0], "bad": probs_list[1]}
        return probs, labels[best_idx]

    
if __name__ == '__main__':
    logger.info(sys.argv)
    img = None
    if len(sys.argv) > 2:
        img = sys.argv[2]
    inference = TorchSimpleInferencer(sys.argv[1])
    probs, label = inference.infer_img(img, .2)
    logger.info("probs {} label {}", probs, label)
