import os
#import torch
#import torch.utils.data as data
import numpy as np
from PIL import Image
#from torchvision.datasets.folder import default_loader
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
# The following two functions are copied from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


classes = [
    "Sky",
    "Building",
    "Column-Pole",
    "Road",
    "Sidewalk",
    "Tree",
    "Sign-Symbol",
    "Fence",
    "Car",
    "Pedestrain",
    "Bicyclist",
    "Void",
]

# can't verify below?
# https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua
""" class_weight = torch.FloatTensor([
    0.58872014284134, 0.51052379608154, 2.6966278553009,
    0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
    2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0]) """
# class_weight = torch.FloatTensor(
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
# )
class_weight = tf.constant(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=tf.float32
)

mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class LabelTensorToPILImage(object):
    def __call__(self, label):
        #label = label.unsqueeze(0)
        label = tf.expand_dims(label, axis=0)
        #colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        colored_label = tf.zeros(shape=(label.shape[0], label.shape[1], 3))
        for i, color in enumerate(class_color):
            mask = tf.math.equal(label, i)
            for j in range(3):
                #colored_label[j].masked_fill_(mask, color[j])
                colored_label[:,:,j] += tf.cast(mask, dtype=tf.float32) * color[j]
        npimg = colored_label.numpy()
        #npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)


class CamVid(Sequence):
    def __init__(
        self,
        root,
        batch_size,
        split="train",
        joint_transform=None,
        transform=None,
        target_transform=None,
        download=False,
        loader=load_img,
    ):
        super(CamVid, self).__init__()
        self.root = root
        self.batch_size = batch_size

        assert split in ("train", "val", "test")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std

        if download:
            self.download()

        # print(type(self.root))
        self.imgs = _make_dataset(os.path.join(self.root, self.split))
        self.data_list = np.array_split(self.imgs, len(self.imgs)//self.batch_size)
        dummy_img = img_to_array(self.loader(self.imgs[0]))
        self.input_shape = dummy_img.shape

    def __getitem__(self, index):
        data_idx = data_list[index]
        imgs = []
        targets = []
        for i in data_idx:
            path = self.data_list[i]
            img = img_to_array(self.loader(path))
            target = Image.open(path.replace(self.split, self.split + "annot"))

            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            imgs.append(img)
            targets.append(target)
        img = tf.concat(imgs, axis=0)
        target = tf.concat(targets, axis=0)

        # print(img.size(), target.size())
        return img, target

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
        raise NotImplementedError
