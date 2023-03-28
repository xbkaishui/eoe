import numpy as np
import torch
from PIL import Image
from loguru import logger as glogger


def convert_snapshot_to_model(model_file, dst):
    load_model = torch.load(model_file)
    model = load_model['net']
    torch.save(model, dst_file)

if __name__ == '__main__':
    # model_file = "/opt/eoe/data/results/log_20230319222526_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it2.pt"
    # dst_file = "/tmp/good.pt"
    model_file = "/opt/eoe/data/results/log_20230326091748_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it4.pt"
    model_file = "/opt/eoe/data/results/log_20230326093643_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it4.pt"
    model_file = "/opt/eoe/data/results/log_20230326103916_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it1.pt"
    # model_file = "/opt/eoe/data/results/log_20230326104613_clip_chip_one_vs_rest_E200/snapshots/snapshot_cls0_it1.pt"
    dst_file = "/opt/eoe/data/models/03_26_good.pt"
    
    # model_file = "/opt/eoe/data/results/log_20230326091748_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls1_it4.pt"
    # dst_file = "/opt/eoe/data/models/03_26_bad.pt"

    # model_file = "/opt/eoe/data/results/log_20230326095038_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it1.pt"
    # dst_file = "/tmp/good_03_26.pt"
    
    model_file = "/opt/eoe/data/results/log_20230328205456_clip_chip_one_vs_rest_E20/snapshots/snapshot_cls0_it0.pt"
    dst_file = "/opt/eoe/data/models/panzi_good.pt"
    convert_snapshot_to_model(model_file, dst_file)