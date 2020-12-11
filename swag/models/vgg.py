"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import tensorflow as tf 
import tensorflow.keras.layers as tkl
#import torch.nn as nn
#import torchvision.transforms as transforms

__all__ = ["VGG16", "VGG16BN", "VGG19", "VGG19BN"]
initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='truncated_normal')
fc_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')

def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [tkl.MaxPool2D(pool_size=(2,2), strides=(2, 2))]
        else:
            conv2d = tkl.Conv2D(v, kernel_size=3, padding='same', kernel_initializer=initializer)
            if batch_norm:
                layers += [conv2d, tkl.BatchNormalization(), tkl.ReLU()]
            else:
                layers += [conv2d, tkl.ReLU()]
            in_channels = v
    return tf.keras.Sequential(layers)


cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(tf.keras.Model):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, input_shape=None):
        super(VGG, self).__init__()

        self.Input = tf.keras.Input(shape=input_shape)
        self.features = make_layers(cfg[depth], batch_norm)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, num_classes),
        # )
        self.classifier = tf.keras.Sequential([
            tkl.Dropout(rate=0.5),
            tkl.Dense(512, kernel_initializer=fc_init, bias_initializer=fc_init),
            tkl.ReLU(),
            tkl.Dropout(rate=0.5),
            tkl.Dense(512, kernel_initializer=fc_init, bias_initializer=fc_init),
            tkl.ReLU(),
            tkl.Dense(num_classes, kernel_initializer=fc_init, bias_initializer=fc_init)
            ])

        out = self.features(self.Input)
        out = tf.keras.layers.Flatten()(out)
        out = self.classifier(out)
        self.model = tf.keras.Model(inputs=self.Input, outputs=out)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2.0 / n))
        #         m.bias.data.zero_()

    def call(self, x):
        x = self.model(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    # transform_train = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.Resize(32),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #         # transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    #     ]
    # )

    # transform_test = transforms.Compose(
    #     [
    #         transforms.Resize(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         # transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
    #     ]
    # )

    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (img/255.0, y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_flip_left_right(img),y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: ((img-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y),
        lambda img, y: (img/255.0, y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225], y)
    ]


class VGG16(Base):
    pass


class VGG16BN(Base):
    kwargs = {"batch_norm": True}


class VGG19(Base):
    kwargs = {"depth": 19}


class VGG19BN(Base):
    kwargs = {"depth": 19, "batch_norm": True}
