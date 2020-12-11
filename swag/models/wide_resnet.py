"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
import tensorflow as tf 
import tensorflow.keras.layers as tkl
import math

__all__ = ["WideResNet28x10"]
initializer = tf.keras.initializers.HeNormal()

def conv3x3(in_planes, out_planes, stride=1):
    return tkl.Conv2D(
        out_planes, kernel_size=3, strides=(stride, stride), padding='same', kernel_initializer=initializer
    )


# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         init.xavier_uniform(m.weight, gain=math.sqrt(2))
#         init.constant(m.bias, 0)
#     elif classname.find("BatchNorm") != -1:
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0)


class WideBasic(tf.keras.layers.Layer):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = tkl.BatchNormalization()
        self.conv1 = tkl.Conv2D(planes, kernel_size=3, padding='same', kernel_initializer=initializer)
        self.dropout = tkl.Dropout(rate=dropout_rate)
        self.bn2 = tkl.BatchNormalization()
        self.conv2 = tkl.Conv2D(
            planes, kernel_size=3, strides=(stride, stride), padding='same', kernel_initializer=initializer
        )

        #self.shortcut = nn.Sequential()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        if stride != 1 or in_planes != planes:
            self.shortcut = tkl.Conv2D(planes, kernel_size=1, strides=(stride,stride), kernel_initializer=initializer)
        self.relu = tf.keras.ReLU()

    def call(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.stride != 1 or self.in_planes != self.planes:
            out += self.shortcut(x)

        return out


class WideResNet(tf.keras.Model):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, dropout_rate=0.0, input_shape=None):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2)
        self.bn1 = tkl.BatchNormalization(momentum=0.9)
        self.linear = tkl.Dense(num_classes, activation='softmax', kernel_initializer=initializer)

        self.Input = tf.keras.Input(shape=input_shape)
        out = self.conv1(self.Input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = tkl.ReLU()(self.bn1(out))
        out = tkl.AveragePool2D((8,8))(out)
        out = tkl.Flatten()(out)
        out = self.linear(out)

        self.model = tf.keras.Model(inputs=self.Input, outputs=out)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return tf.keras.Sequential(layers)

    def call(self, x):
        out = self.model(x)

        return out


class WideResNet28x10:
    base = WideResNet
    args = list()
    kwargs = {"depth": 28, "widen_factor": 10}
    # transform_train = transforms.Compose(
    #     [
    #         transforms.Resize(32),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    # transform_test = transforms.Compose(
    #     [
    #         transforms.Resize(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (img/255.0, y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (img/255.0, y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
