"""
    100-layer tiramisu/fc densenet67 model definition
    ported from: #from: https://github.com/bfortuner/pytorch_tiramisu
"""

# import torch
# import torch.nn as nn
# from torchvision import transforms

import tensorflow as tf 
import tensorflow.keras.layers as tkl

from .layers import DenseBlock, TransitionDown, TransitionUp, Bottleneck

# import .joint_transforms as joint_transforms
from .joint_transforms import (
    JointRandomResizedCrop,
    JointRandomHorizontalFlip,
    JointCompose,
    LabelToLongTensor,
)

__all__ = ["FCDenseNet57", "FCDenseNet67", "FCDenseNet103"]


class FCDenseNet(tf.keras.Model):
    def __init__(
        self,
        in_channels=3,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        bottleneck_layers=5,
        growth_rate=16,
        out_chans_first_conv=48,
        num_classes=11,
        use_aleatoric=False,
        input_shape=None
    ):
        super(FCDenseNet, self).__init__()

        self.use_aleatoric = use_aleatoric
        self.num_classes = num_classes

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.Input = tf.keras.Input(shape=input_shape)

        self.firstconv=tkl.Conv2D(out_chans_first_conv, kernel_size=3, strides=(1,1), padding='same')

        # self.add_module(
        #     "firstconv",
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=out_chans_first_conv,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=True,
        #     ),
        # )
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = tf.keras.Sequential()
        self.transDownBlocks = tf.keras.Sequential()
        for i in range(len(down_blocks)):
            self.denseBlocksDown.add(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i])
            )
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.bottleneck = Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)
        # self.add_module(
        #     "bottleneck", Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)
        # )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = tf.keras.Sequential()
        self.denseBlocksUp = tf.keras.Sequential()
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.add(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.add(
                DenseBlock(cur_channels_count, growth_rate, up_blocks[i], upsample=True)
            )
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.add(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.add(
            DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False)
        )
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Softmax ##

        if use_aleatoric:
            final_out_channels = num_classes * 2
        else:
            final_out_channels = num_classes

        # self.finalConv = nn.Conv2d(
        #     in_channels=cur_channels_count,
        #     out_channels=final_out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=True,
        # )
        self.finalConv = tkl.Conv2D(final_out_channels, kernel_size=1, strides=(1,1), padding='same')
        # self.log_softmax = nn.LogSoftmax(dim=1)

        out = self.firstconv(self.Input)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        if self.use_aleatoric:
            out = tf.split(out, 2, 3)
            out = tf.concat([tf.expand_dims(i, axis=3) for i in out], 3)

        self.model = tf.keras.Model(inputs=self.Input, outputs=out)

    def call(self, x):
        # out = self.firstconv(x)

        # skip_connections = []
        # for i in range(len(self.down_blocks)):
        #     out = self.denseBlocksDown[i](out)
        #     skip_connections.append(out)
        #     out = self.transDownBlocks[i](out)

        # out = self.bottleneck(out)
        # for i in range(len(self.up_blocks)):
        #     skip = skip_connections.pop()
        #     out = self.transUpBlocks[i](out, skip)
        #     out = self.denseBlocksUp[i](out)

        # out = self.finalConv(out)

        # # output contains both mean and variance so split them
        # if self.use_aleatoric:
        #     out = torch.split(out, self.num_classes, 1)
        #     out = torch.cat([i.unsqueeze(1) for i in out], 1)

        # out = self.log_softmax(out)
        out = self.model(x)
        return out


class FCDenseNet57:
    base = FCDenseNet
    args = list()
    kwargs = {
        "in_channels": 3,
        "down_blocks": (4, 4, 4, 4, 4),
        "up_blocks": (4, 4, 4, 4, 4),
        "bottleneck_layers": 4,
        "growth_rate": 12,
        "out_chans_first_conv": 48,
    }

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    # transform_train = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=camvid_mean, std=camvid_std)]
    # )
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=camvid_mean, std=camvid_std)]
    # )

    joint_transform = JointCompose(
        [
            JointRandomResizedCrop(224),  # commented for fine-tuning
            JointRandomHorizontalFlip(),
        ]
    )
    ft_joint_transform = JointCompose([JointRandomHorizontalFlip()])
    target_transform = LabelToLongTensor()

    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img: (img-camvid_mean)/camvid_std
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        lambda img: (img-camvid_mean)/camvid_std
    ]

class FCDenseNet67:
    base = FCDenseNet
    args = list()

    kwargs = {
        "in_channels": 3,
        "down_blocks": (5, 5, 5, 5, 5),
        "up_blocks": (5, 5, 5, 5, 5),
        "bottleneck_layers": 5,
        "growth_rate": 16,
        "out_chans_first_conv": 48,
    }

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    # transform_train = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    #     ]
    # )
    # transform_test = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    #     ]
    # )

    joint_transform = JointCompose(
        [
            JointRandomResizedCrop(224),  # commented for fine-tuning
            JointRandomHorizontalFlip(),
        ]
    )
    ft_joint_transform = JointCompose([JointRandomHorizontalFlip()])

    target_transform = LabelToLongTensor()

    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        #lambda img: (img-camvid_mean)/camvid_std
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        #lambda img: (img-camvid_mean)/camvid_std
    ]

class FCDenseNet103:
    base = FCDenseNet
    args = list()

    kwargs = {
        "in_channels": 3,
        "down_blocks": (4, 5, 7, 10, 12),
        "up_blocks": (12, 10, 7, 5, 4),
        "bottleneck_layers": 15,
        "growth_rate": 16,
        "out_chans_first_conv": 48,
    }

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    # transform_train = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    #     ]
    # )
    # transform_test = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    #     ]
    # )

    joint_transform = JointCompose(
        [
            JointRandomResizedCrop(224),  # commented for fine-tuning
            JointRandomHorizontalFlip(),
        ]
    )
    ft_joint_transform = JointCompose([JointRandomHorizontalFlip()])

    target_transform = LabelToLongTensor()

    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        #lambda img: (img-camvid_mean)/camvid_std
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (tf.image.per_image_standardization(img), y),
        #lambda img: (img-camvid_mean)/camvid_std
    ]
