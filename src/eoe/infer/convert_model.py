import numpy as np
import torch
from PIL import Image
from loguru import logger as glogger


def convert_snapshot_to_model(model_file, dst):
    load_model = torch.load(model_file)
    model = load_model['net']
    torch.save(model, dst_file)

if __name__ == '__main__':
    model_file = "/opt/eoe/data/results/log_20230319222526_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it2.pt"
    dst_file = "/tmp/good.pt"
    convert_snapshot_to_model(model_file, dst_file)