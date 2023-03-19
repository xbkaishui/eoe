import torch
import torchvision


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


mnist = torchvision.datasets.MNIST('/tmp/datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
# print(mnist)
# img, target = mnist.__getitem__(0)
# print(img)

from eoe.datasets.chip import CHIP

chip = CHIP("/Users/xbkaishui/Downloads/chip_clip_dect")
img, target, index = chip.__getitem__(0)
print(img.shape)

from eoe.datasets.cifar import CIFAR10

cifar10 = CIFAR10('/tmp/datasets/', train=True, download=True)
img, target, index = cifar10.__getitem__(0)
print(img.shape)
print(type(img))