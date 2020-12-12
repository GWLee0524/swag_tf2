import numpy as np
import torch
import torchvision
import os
import tensorflow as tf 
from tensorflow.keras.utils import Sequence, SequenceEnqueuer 
import tensorflow_datasets as tfds


from .camvid import CamVid

c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)


def camvid_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
    joint_transform=None,
    ft_joint_transform=None,
    ft_batch_size=1,
    **kwargs
):

    # load training and finetuning datasets
    print(path)
    train_set = CamVid(
        root=path,
        split="train",
        joint_transform=joint_transform,
        transform=transform_train,
        **kwargs
    )
    ft_train_set = CamVid(
        root=path,
        split="train",
        joint_transform=ft_joint_transform,
        transform=transform_train,
        **kwargs
    )

    val_set = CamVid(
        root=path, split="val", joint_transform=None, transform=transform_test, **kwargs
    )
    test_set = CamVid(
        root=path,
        split="test",
        joint_transform=None,
        transform=transform_test,
        **kwargs
    )
    input_shape = train_set.input_shape
    num_classes = 11  # hard coded labels ehre
    train_loader = SequenceEnqueuer(train_set)
    ft_train_loader = SequenceEnqueuer(ft_train_set)
    val_loader = SequenceEnqueuer(val_set)
    test_loader = SequenceEnqueuer(test_set)
    
    train_loader.start()
    ft_train_loader.start()
    val_loader.start()
    test_loader.start()

    # return (
    #     {
    #         "train": torch.utils.data.DataLoader(
    #             train_set,
    #             batch_size=batch_size,
    #             shuffle=shuffle_train,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #         "fine_tune": torch.utils.data.DataLoader(
    #             ft_train_set,
    #             batch_size=ft_batch_size,
    #             shuffle=shuffle_train,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #         "val": torch.utils.data.DataLoader(
    #             val_set,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #         "test": torch.utils.data.DataLoader(
    #             test_set,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #     },
    #     num_classes,
    # )

    return (
        {
            "train": train_loader.get(),
            "fine_tune": ft_train_loader.get(),
            "val": val_loader.get(),
            "test": test_loader.get()
        },
        num_classes, input_shape
        )


def svhn_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
):
    # train_set = torchvision.datasets.SVHN(
    #     root=path, split="train", download=True, transform=transform_train
    # )
    import tensorflow_datasets as tfds

    if use_validation:
        # test_set = torchvision.datasets.SVHN(
        #     root=path, split="train", download=True, transform=transform_test
        # )
        # train_set.data = train_set.data[:-val_size]
        # train_set.labels = train_set.labels[:-val_size]

        # test_set.data = test_set.data[-val_size:]
        # test_set.labels = test_set.labels[-val_size:]

        train_set, info = tfds.load('svhn_cropped', split='train[0:-%d]'%(val_size), with_info=True)
        test_set = tfds.load('svhn_cropped', split='train[val_size:]'%(val_size))

    else:
        print("You are going to run models on the test set. Are you sure?")
        # test_set = torchvision.datasets.SVHN(
        #     root=path, split="test", download=True, transform=transform_test
        # )
        train_set, info = tfds.load('svhn_cropped', split='train', with_info=True)
        test_set = tfds.load('svhn_cropped', split='test')
        input_shape = info.features['image'].shape
    num_classes = 10
    for t in transform_train:
        train_set = train_set.map(t)
    for t in transform_test:
        test_set = test_set.map(t)

    train_set = train_set.batch(batch_size)
    test_set = test_set.batch(batch_size)
    #train_set = train_set.batch(batch_size).map(lambda x, y:(x, tf.one_hot(y, depth=num_classes)))
    #test_set = test_set.batch(batch_size).map(lambda x, y:(x, tf.one_hot(y, depth=num_classes)))
    
    return (
        {
            "train": tfds.as_numpy(train_set),
            "test": tfds.as_numpy(test_set),
        },
        num_classes, input_shape
    )
    # return (
    #     {
    #         "train": torch.utils.data.DataLoader(
    #             train_set,
    #             batch_size=batch_size,
    #             shuffle=True and shuffle_train,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #         "test": torch.utils.data.DataLoader(
    #             test_set,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #     },
    #     num_classes,
    # )



def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation=True,
    val_size=5000,
    split_classes=None,
    shuffle_train=True,
    **kwargs
):

    if dataset == "CamVid":
        return camvid_loaders(
            path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_train=transform_train,
            transform_test=transform_test,
            use_validation=use_validation,
            val_size=val_size,
            **kwargs
        )

    path = os.path.join(path, dataset.lower())

    #ds = getattr(torchvision.datasets, dataset)

    if dataset == "SVHN":
        return svhn_loaders(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            use_validation,
            val_size,
        )
    # else:
    #     #ds = getattr(torchvision.datasets, dataset)

    if dataset == "STL10":
        # train_set = ds(
        #     root=path, split="train", download=True, transform=transform_train
        # )
        train_set, info = tfds.load('stl10', split='train', as_supervised=True, with_info=True, shuffle_files=shuffle_train)
        num_classes = info.features['label'].num_classes
        num_train = info.splits['train'].num_examples
        input_shape = info.features['image'].shape
        #cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        #train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set, info = tfds.load(dataset.lower(), split='train', as_supervised=True, with_info=True, shuffle_files=shuffle_train)
        num_classes = info.features['label'].num_classes
        num_train = info.splits['train'].num_examples
        input_shape = info.features['image'].shape
        #num_classes = max(train_set.targets) + 1

    if use_validation:
        print(
            "Using train ("
            + str(num_train - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )
        # train_set.data = train_set.data[:-val_size]
        # train_set.targets = train_set.targets[:-val_size]

        # test_set = ds(root=path, train=True, download=True, transform=transform_test)
        # test_set.train = False
        # test_set.data = test_set.data[-val_size:]
        # test_set.targets = test_set.targets[-val_size:]

        train_set = tfds.load(dataset.lower(), split='train[:-%d]'%(val_size), as_supervised=True, shuffle_files=shuffle_train)
        test_set = tfds.load(dataset.lower(), split='test[-%d:]'%(val_size), as_supervised=True)

        # delattr(test_set, 'data')
        # delattr(test_set, 'targets')
    else:
        print("You are going to run models on the test set. Are you sure?")
        if dataset == "STL10":
            # test_set = ds(
            #     root=path, split="test", download=True, transform=transform_test
            # )
            # test_set.labels = cls_mapping[test_set.labels]
            test_set = tfds.load(dataset.lower(), split='test', as_supervised=True)
        else:
            # test_set = ds(
            #     root=path, train=False, download=True, transform=transform_test
            # )
            test_set = tfds.load(dataset.lower(), split='test', as_supervised=True)

    if split_classes is not None:
        assert dataset == "CIFAR10"
        assert split_classes in {0, 1}

        print("Using classes:", end="")
        print(c10_classes[split_classes])

        def filter_func(x, y):
            allowed_labels = c10_classes[split_classes]
            label = y
            isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
            reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
            return tf.greater(reduced, tf.constant(0.))
        train_set.filter(filter_func)
        test_set.filter(filter_func)

        # train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        # train_set.data = train_set.data[train_mask, :]
        # train_set.targets = np.array(train_set.targets)[train_mask]
        # train_set.targets = np.where(
        #     train_set.targets[:, None] == c10_classes[split_classes][None, :]
        # )[1].tolist()
        # print("Train: %d/%d" % (train_set.data.shape[0], train_mask.size))

        # test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        # print(test_set.data.shape, test_mask.shape)
        # test_set.data = test_set.data[test_mask, :]
        # test_set.targets = np.array(test_set.targets)[test_mask]
        # test_set.targets = np.where(
        #     test_set.targets[:, None] == c10_classes[split_classes][None, :]
        # )[1].tolist()
        # print("Test: %d/%d" % (test_set.data.shape[0], test_mask.size))
    for t in transform_train:
        train_set = train_set.map(t)
    for t in transform_test:
        test_set = test_set.map(t)

    #train_set = train_set.batch(batch_size).map(lambda x, y:(x, tf.one_hot(y, depth=num_classes)))
    #test_set = test_set.batch(batch_size).map(lambda x, y:(x, tf.one_hot(y, depth=num_classes)))
    train_set = train_set.batch(batch_size)
    test_set = test_set.batch(batch_size)
    return (
        {
            "train": tfds.as_numpy(train_set),
            "test": tfds.as_numpy(test_set)
        },
        num_classes, input_shape
        )
    # return (
    #     {
    #         "train": torch.utils.data.DataLoader(
    #             train_set,
    #             batch_size=batch_size,
    #             shuffle=True and shuffle_train,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #         "test": torch.utils.data.DataLoader(
    #             test_set,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         ),
    #     },
    #     num_classes,
    # )
