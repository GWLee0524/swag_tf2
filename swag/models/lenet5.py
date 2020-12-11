import math
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms

import tensorflow as tf 
import tensorflow_transform as tft

# import bnn

__all__ = ["LeNet5"]
initializer = tf.keras.initializers.HeNormal()

class LeNet5Base(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super(LeNet5Base, self).__init__()
        # self.conv_part = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2),
        # )
        # self.fc_part = nn.Sequential(
        #     nn.Linear(800, 500), nn.ReLU(True), nn.Linear(500, num_classes.item())
        # )

        # Initialize weights
        # for m in self.conv_part.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2.0 / n))
        #         m.bias.data.zero_()
        self.input_shape = input_shape
        self.Input = tf.keras.Input(shape=self.input_shape)
        self.num_classes = num_classes
        
        
        self.model_seq = tf.keras.Sequential()
        self.model_seq.add(tf.keras.layers.Conv2D(20, kernel_size=5, strides=(1,1), padding='valid', activation='relu', kernel_initializer=initializer))
        self.model_seq.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model_seq.add(tf.keras.layers.Conv2D(50, kernel_size=5, strides=(1,1), padding='valid', activation='relu', kernel_initializer=initializer))
        self.model_seq.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model_seq.add(tf.keras.layers.Flatten())
        self.model_seq.add(tf.keras.layers.Dense(500, activation='relu', kernel_initializer=initializer))
        self.model_seq.add(tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer=initializer))
        out = self.model_seq(self.Input)
        out = tf.sequeeze(out)
        self.model = tf.keras.Model(inputs=self.Input, outputs=out)

    def forward(self, x):
        # x = self.conv_part(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc_part(x)

        x = self.model(x)
        return x


"""class LeNet5BNN(bnn.BayesianModule):

    def __init__(self, num_classes):
        super(LeNet5BNN, self).__init__()
        self.conv_part = nn.Sequential(
            bnn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            bnn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_part = nn.Sequential(
            bnn.Linear(800, 500),
            nn.ReLU(True),
            bnn.Linear(500, num_classes)
        )

        # Initialize weights

        for m in self.modules():
            if isinstance(m, bnn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mean.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x"""


class LeNet5:
    base = LeNet5Base
    # bnn = LeNet5BNN
    args = list()
    kwargs = {}

    # transform_train = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(28, padding=4),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    #     ]
    # )

    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    # )

    # transform_train = [
    #     lambda img: tf.image.random_flip_left_right(x),
    #     lambda img: tf.image.random_crop(x[:, 4:img.shape[1]-4, 4:img.shape[2]-4,:], 28),
    #     lambda img: tf.conver_to_tensor(img),
    #     lambda img: (img-0.1307)/0.3081
    # ]
    # transform_test = [
    #     lambda img: tf.conver_to_tensor(img),
    #     lambda img: (img-0.1307)/0.3081
    # ]
    transform_train = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (img/255.0, y),
        lambda img, y: (tf.image.random_flip_left_right(x), y),
        lambda img, y: (tf.image.random_crop(img, [28, 28, 3]), y),
        lambda img, y: ((img-0.1307)/0.3081, y)
    ]
    transform_test = [
        lambda img, y: (tf.cast(tf.convert_to_tensor(img), dtype=tf.float32), y) ,
        lambda img, y: (img/255.0, y),
        lambda img, y: ((img-0.1307)/0.3081, y)
    ]
