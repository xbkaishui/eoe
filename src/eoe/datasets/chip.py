from typing import List, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import torchvision
from loguru import logger as glogger


class ADCHIP(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """ AD dataset for MNIST. Implements :class:`eoe.datasets.bases.TorchvisionDataset`. """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 10, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )
        glogger.info("self root {}, normal_classes {}, nominal_label {}", root, normal_classes, nominal_label)

        self._train_set = CHIP(
            self.root, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = CHIP(
            root=self.root, transform=self.test_transform,
            target_transform=self.target_transform, conditional_transform=self.test_conditional_transform
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = CHIP(
            self.root,
            transform=transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class CHIP(torchvision.datasets.vision.VisionDataset):
    def __init__(self, *args, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's MNIST s.t. it handles the optional conditional transforms.
        See :class:`eoe.datasets.bases.TorchvisionDataset`. Apart from this, the implementation doesn't differ from the
        standard one.
        """
        super(CHIP, self).__init__(*args, **kwargs)
        glogger.info("root dir {}", self.root)
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        images_dir = Path(self.root) / 'chip' / 'images'
        labels_dir = Path(self.root) / 'chip' / 'labels'
        self.images = [n for n in images_dir.iterdir()]
        # self.labels = []
        self.targets = []
        for image in self.images:
            base, _ = os.path.splitext(os.path.basename(image))
            label = labels_dir / f'{base}.txt'
            # self.labels.append(label if label.exists() else None)
            with open(label, 'r') as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
                # one line one label
                target = int(labels[0][0])
                self.targets.append(target)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        img = Image.open(self.images[index]).convert('RGB')
        img = np.asarray(img)
        # img, target = self.data[index], self.targets[index]
        target = self.targets[index]
        
        if self.transform is None or isinstance(self.transform, transforms.Compose) and len(self.transform.transforms) == 0:
            img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
        else:
            # not reach this
            img = Image.fromarray(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)
        return img, target, index

