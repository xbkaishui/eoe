from PIL import Image
import requests
from loguru import logger
import torch

image = Image.open("example_image.jpg")
logger.info("image {}", image)

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
logger.info("load model")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# print cost time
logger.info("load model done")
prompts = ["a glass", "something to fill", "wood", "a jar"]

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

import torch
import matplotlib.pyplot as plt

logger.info("predict")
# predict
with torch.no_grad():
  outputs = model(**inputs)

preds = outputs.logits.unsqueeze(1)

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)];
[ax[i+1].text(0, -15, prompts[i]) for i in range(4)];

filename = f"mask.png"
# here we save the second mask
logger.info("save mask to {}", filename)
plt.imsave(filename,torch.sigmoid(preds[1][0]))