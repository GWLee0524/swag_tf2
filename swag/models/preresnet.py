"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

#import torch.nn as nn
#import torchvision.transforms as transforms

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow.image as tfi
import math

__all__ = ["PreResNet110", "PreResNet56", "PreResNet8", "PreResNet83", "PreResNet164"]

#initializer = tf.keras.initializers.HeNormal()
initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out')
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
#     )


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample
        self.stride = (stride, stride)
        self.inplanes = inplanes
        self.planes = planes

        self.relu = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', kernel_initializer=initializer)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same',kernel_initializer=initializer)

    def call(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # out = tf.keras.layers.BatchNormalization()(x)
        # out = tf.keras.layers.ReLU()(out)
        # out = tf.keras.layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', kernel_initializer=initializer)(out)
        
        # out = tf.keras.layers.BatchNormalization()(out)
        # out = tf.keras.layers.ReLU()(out)
        # out = tf.keras.layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same',kernel_initializer=initializer)(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.bn1 = nn.BatchNorm2d(inplanes)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(
        #     planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        # )
        # self.bn3 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = (stride, stride)
        self.inplanes = inplanes
        self.planes = planes
        
        self.relu = tf.keras.layers.ReLU()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(self.planes, kernel_size=1, strides=(1,1), padding='same', kernel_initializer=initializer)

        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', kernel_initializer=initializer)

        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(self.planes*4, kernel_size=1, strides=(1,1), padding='same', kernel_initializer=initializer)


    def call(self, x):
        residual = x

        # out = self.bn1(x)
        # out = self.relu(out)
        # out = self.conv1(out)

        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.conv2(out)

        # out = self.bn3(out)
        # out = self.relu(out)
        # out = self.conv3(out)
        out = self.batchnorm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(tf.keras.Model):
    def __init__(self, num_classes=10, depth=110, input_shape=None):
        super(PreResNet, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, "depth should be 9n+2"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, "depth should be 6n+2"
            n = (depth - 2) // 6
            block = BasicBlock

        self.inplanes = 16
        self.num_classes = num_classes
        self.Input = tf.keras.Input(shape=input_shape)
        self.model = tf.keras.Sequential()
        self.model.add(self.Input)
        self.model.add(tf.keras.layers.Conv2D(self.inplanes, kernel_size=3, strides=(1,1), padding='same', use_bias=False, kernel_initializer=initializer))
        #self.model.add(self._make_layer(block, 16, n))
        for l in self._make_layer(block, 16, n):
            self.model.add(l)
        #self.model.add(self._make_layer(block, 32, n, stride=2))
        for l in self._make_layer(block, 32, n, stride=2):
            self.model.add(l)
        #self.model.add(self._make_layer(block, 64, n, stride=2))
        for l in self._make_layer(block, 64, n, stride=2):
            self.model.add(l)
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(8,8)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer=initializer))
        
        # self.bn = nn.BatchNorm2d(64 * block.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2.0 / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(
            #         self.inplanes,
            #         planes * block.expansion,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     )
            # )
            downsample = tf.keras.layers.Conv2D(planes * block.expansion, kernel_size=1, \
                strides=(stride, stride), use_bias=False, kernel_initializer=initializer)

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def call(self, x):
        x = self.model(x)

        # x = self.layer1(x)  # 32x32
        # x = self.layer2(x)  # 16x16
        # x = self.layer3(x)  # 8x8
        # x = self.bn(x)
        # x = self.relu(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class PreResNet164:
    base = PreResNet
    args = list()
    kwargs = {"depth": 164}
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
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]


class PreResNet110:
    base = PreResNet
    args = list()
    kwargs = {"depth": 110}
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
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]

class PreResNet83:
    base = PreResNet
    args = list()
    kwargs = {"depth": 83}
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
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]


class PreResNet56:
    base = PreResNet
    args = list()
    kwargs = {"depth": 56}
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
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]


class PreResNet8:
    base = PreResNet
    args = list()
    kwargs = {"depth": 8}
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
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: (tf.image.random_crop(img, [32, 32, 3]), y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img, y: (tf.image.resize(img, [32, 32]), y),
        lambda img, y: ((img-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010], y)
    ]
