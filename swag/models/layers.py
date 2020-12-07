"""
    layer definitions for 100-layer tiramisu
    #from: https://github.com/bfortuner/pytorch_tiramisu
"""
# import torch
# import torch.nn as nn

import tensorflow as tf 


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        
        # self.add_module("norm", nn.BatchNorm2d(in_channels))
        # self.add_module("relu", nn.ReLU(True))
        # self.add_module(
        #     "conv",
        #     nn.Conv2d(
        #         in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True
        #     ),
        # )
        # self.add_module("drop", nn.Dropout(p=0.2))

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d = tf.keras.layers.Conv2d(self.growth_rate, kernel_size=3, strides=(1,1), padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

    def call(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.dropout(x)

        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        # self.layers = nn.ModuleList(
        #     [
        #         DenseLayer(in_channels + i * growth_rate, growth_rate)
        #         for i in range(n_layers)
        #     ]
        # )
        #self.Input = tf.keras.Input(shape=(None, in_channels))
        self.layers = [DenseLayer(in_channels + i * growth_rate, growth_rate)
                for i in range(n_layers)]
            

    def call(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            # for layer in self.layers:
            #     out = layer(x)
            #     x = torch.cat([x, out], 1)
            #     new_features.append(out)

            for layer in self.layers:
                out = layer(x)
                x = tf.concat([x, out], axis=3)
                new_features.append(out)
            return tf.concat(new_features, axis=3)
        else:
            # for layer in self.layers:
            #     out = layer(x)
            #     x = torch.cat([x, out], 1)  # 1 = channel axis
            for layer in self.layers:
                out = layer(x)
                x = tf.concat([x, out], axis=3)
            return x


class TransitionDown(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        
        # self.add_module("norm", nn.BatchNorm2d(num_features=in_channels))
        # self.add_module("relu", nn.ReLU(inplace=True))
        # self.add_module(
        #     "conv",
        #     nn.Conv2d(
        #         in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True
        #     ),
        # )
        # self.add_module("drop", nn.Dropout2d(0.2))
        # self.add_module("maxpool", nn.MaxPool2d(2))

        self.in_channels = in_channels
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d = tf.keras.layers.Conv2D(self.in_channels, kernel_size=1, strides=(1,1), padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2,2))

    def call(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        return x


class TransitionUp(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(self.out_channels, kerenl_size=3, strides=(2,2), padding='valid')

    def call(self, x, skip):
        # out = self.convTrans(x)
        # out = center_crop(out, skip.size(2), skip.size(3))
        # out = torch.cat([out, skip], 1)

        out = self.conv_transpose(x)
        out = center_crop(out, skip.shape[1], skip.shape[2])
        out = tf.concat([out, skip], axis=3)

        return out


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module(
            "bottleneck", DenseBlock(in_channels, growth_rate, n_layers, upsample=True)
        )
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.layer = DenseBlock(in_channels, growth_rate, n_layers, upsample=True)
    def call(self, x):
        return self.layer(x)
        #return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, h, w, _ = layer.shape #tensorflow batch, height, width, channels
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width), :]
