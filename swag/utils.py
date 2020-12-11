import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import tensorflow as tf 
#import torch.nn.functional as F


def flatten(lst):
    #tmp = [i.contiguous().view(-1, 1) for i in lst]
    tmp = [tf.reshape(i, (-1, 1)) for i in lst]
    return tf.squeeze(tf.concat(tmp, axis=0))
    #return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        #n = tensor.numel()
        #outList.append(vector[:, i : i + n].view(tensor.shape))
        n = tf.size(tensor)
        outList.append(tf.reshape(vector[:, i: i+n], shape=tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    #m, _ = torch.max(x, dim=dim, keepdim=True)
    m = tf.keras.backend.max(x, axis=dim, keepdims=True)
    return m + tf.math.log( tf.math.reduce_sum(tf.math.exp(x-m), axis=dim, keepdims=True) )
    #return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    optimizer.learning_rate = lr
    return lr


# def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
#     state = {"epoch": epoch}
#     state.update(kwargs)
#     #filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
#     filepath = os.path.join(dir, "%s-%d_cpkt" % (name, epoch))
#     state['model'].save(filepath)
#     state['optimizer']
#     torch.save(state, filepath)

def set_velocity(params):
    velocity = []
    for p in params:
        v = tf.zeros_like(p)
        velocity.append(v)
    return velocity

def apply_gradient_with_vel(train_vars, grads, velocity, lr, momentum):
    #velocity = momentum * velocity + grads
    new_velocity = []
    for var, g, vel in zip(train_vars, grads, velocity):
        vel = momentum * vel + g
        var.assign(var - lr * vel)
        new_velocity.append(vel)
    return train_vars, new_velocity

def train_epoch_v2(
    loader,
    model,
    criterion,
    optimizer,
    weight_decay=1e-4,
    regression=False,
    verbose=False,
    subset=None,
    velocity=None
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader) if "__len__" in dir(loader) else len(list(loader))

    #model.train()
    regularizer = tf.keras.regularizers.L2(l2=weight_decay)

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        # if cuda:
        #     input = input.cuda(non_blocking=True)
        #     target = target.cuda(non_blocking=True)
        #print(input[0])
        with tf.GradientTape() as tape:
            input = tf.Variable(input, dtype=tf.float32)
            train_vars = model.trainable_variables
            tape.watch(train_vars)
            loss, output = criterion(model, input, target)
            for var in train_vars:
                loss += regularizer(var)
        grads = tape.gradient(loss, train_vars)

        if velocity is None:
            velocity=set_velocity(train_vars)

        train_vars, velocity = apply_gradient_with_vel(train_vars, grads, velocity, optimizer.lr, optimizer.momentum)

        #optimizer.apply_gradients(zip(grads, train_vars))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        #loss_sum += loss.data.item() * input.size(0)
        loss_sum += loss.numpy().item() * input.shape[0]

        if not regression:
            #pred = output.data.argmax(1, keepdim=True)
            #correct += pred.eq(target.data.view_as(pred)).sum().item()
            
            pred = tf.math.argmax(output, 1)
            target = tf.math.argmax(target, 1)
            correct += tf.math.reduce_sum(tf.cast(tf.math.equal(pred, target), dtype=tf.float32)).numpy().item()

        #num_objects_current += input.size(0)
        num_objects_current += input.shape[0]

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }, velocity

def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    weight_decay=1e-4,
    regression=False,
    verbose=False,
    subset=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader) if "__len__" in dir(loader) else len(list(loader))

    #model.train()
    regularizer = tf.keras.regularizers.L2(l2=weight_decay)

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        # if cuda:
        #     input = input.cuda(non_blocking=True)
        #     target = target.cuda(non_blocking=True)
        #print(input[0])
        with tf.GradientTape() as tape:
            input = tf.Variable(input, dtype=tf.float32)
            train_vars = model.trainable_variables
            tape.watch(train_vars)
            loss, output = criterion(model, input, target)
            for var in train_vars:
                loss += regularizer(var)
        grads = tape.gradient(loss, train_vars)

        optimizer.apply_gradients(zip(grads, train_vars))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        #loss_sum += loss.data.item() * input.size(0)
        loss_sum += loss.numpy().item() * input.shape[0]

        if not regression:
            #pred = output.data.argmax(1, keepdim=True)
            #correct += pred.eq(target.data.view_as(pred)).sum().item()
            
            pred = tf.math.argmax(output, 1)
            target = tf.math.argmax(target, 1)
            correct += tf.math.reduce_sum(tf.cast(tf.math.equal(pred, target), dtype=tf.float32)).numpy().item()

        #num_objects_current += input.size(0)
        num_objects_current += input.shape[0]

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


def eval(loader, model, criterion, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    #num_objects_total = len(loader.dataset)
    #num_objects_total = len(loader) if "__len__" in dir(loader) else len(list(loader))
    num_objects_total = 0.0

    #model.eval()

    # with torch.no_grad():
    #     if verbose:
    #         loader = tqdm.tqdm(loader)
    #     for i, (input, target) in enumerate(loader):
    #         if cuda:
    #             input = input.cuda(non_blocking=True)
    #             target = target.cuda(non_blocking=True)

    #         loss, output = criterion(model, input, target)

    #         loss_sum += loss.item() * input.size(0)

    #         if not regression:
    #             pred = output.data.argmax(1, keepdim=True)
    #             correct += pred.eq(target.data.view_as(pred)).sum().item()


    if verbose:
        loader = tqdm.tqdm(loader)
    for i, (input, target) in enumerate(loader):

        loss, output = criterion(model, input, target)

        loss_sum += loss.numpy().item() * input.shape[0]
        num_objects_total += input.shape[0]
        if not regression:
            # pred = output.data.argmax(1, keepdim=True)
            # correct += pred.eq(target.data.view_as(pred)).sum().item()
            pred = tf.math.argmax(output, 1)
            target = tf.math.argmax(target, 1) #onehot to label
            correct += tf.math.reduce_sum(tf.cast(tf.math.equal(pred, target),dtype=tf.float32)).numpy().item()

    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": None if regression else correct / num_objects_total * 100.0,
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    #model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    # with torch.no_grad():
    #     for input, target in loader:
    #         input = input.cuda(non_blocking=True)
    #         output = model(input)

    #         batch_size = input.size(0)
    #         predictions.append(F.softmax(output, dim=1).cpu().numpy())
    #         targets.append(target.numpy())
    #         offset += batch_size

    for input, target in loader:
            #input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.shape[0]
            #predictions.append(tf.nn.softmax(output, axis=1).numpy())
            predictions.append(output.numpy())
            targets.append(target)
            offset += batch_size
    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def moving_average(net1, net2, alpha=1):
    # for param1, param2 in zip(net1.parameters(), net2.parameters()):
    #     param1.data *= 1.0 - alpha
    #     param1.data += param2.data * alpha
    # TO CHECK: may be deprecated, because if we use tf.train.MovingAverage
    for param1, param2 in zip(net1.trainable_variables, net2.trainable_variables):
        tmp = param1 * (1.0-alpha)
        tmp = tmp + param2 * alpha
        param1.assign(tmp)


# def _check_bn(module, flag):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         flag[0] = True


# def check_bn(model):
#     flag = [False]
#     model.apply(lambda module: _check_bn(module, flag))
#     return flag[0]

def check_bn(model):
    layers = model.trainable_variables
    flag = False
    for layer in layers:
        if 'batch_normalization' in layer.name:
            flag=True
            break
    return flag 

# def reset_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.running_mean = torch.zeros_like(module.running_mean)
#         module.running_var = torch.ones_like(module.running_var)

def reset_bn(model):
    layers = model.trainable_variables
    for layer in layers:
        # gamma reset
        if 'batch_normalization' in layer.name.lower() and 'gamma' in layer.name.lower():
            layer.assign(tf.ones_like(layer))
        # beta
        elif 'batch_normalization' in layer.name.lower() and 'beta' in layer.name.lower():
            layer.assign(tf.zeros_like(layer))


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def get_bn_layer(layer):
    is_layers = False
    bn_layers = []
    for att in dir(layer):
        if 'layers' in att:
            is_layers = True
            break
    if is_layers:
        for l in layer.layers:
            bn_layer = get_bn_layers(l)
            if bn_layer is not None:
                bn_layers += bn_layer #concatenating
        return bn_layers
    else:
        if 'batch_normalization' in layer.name.lower():
            return [layer]
        else:
            return None

def get_all_bn_layers(model):
    layers = model.layers
    bn_layers=[]
    for layer in layers:
        bn_layer = get_bn_layer(layer)
        bn_layers += bn_layer
    return bn_layers



def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader) if "__len__" in dir(loader) else len(list(loader))

    # with torch.no_grad():
    #     if subset is not None:
    #         num_batches = int(num_batches * subset)
    #         loader = itertools.islice(loader, num_batches)
    #     if verbose:

    #         loader = tqdm.tqdm(loader, total=num_batches)
    #     for input, _ in loader:
    #         input = input.cuda(non_blocking=True)
    #         input_var = torch.autograd.Variable(input)
    #         b = input_var.data.size(0)

    #         momentum = b / (n + b)
    #         for module in momenta.keys():
    #             module.momentum = momentum

    #         model(input_var, **kwargs)
    #         n += b
    bn_layers = get_all_bn_layers(model)
    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)
    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)
    for input, _ in loader:
        input_var = tf.Variable(input)
        b = input_var.shape[0]

        momentum = b / (n + b)
        # for module in momenta.keys():
        #     module.momentum = momentum
        for bn in bn_layers:
            bn.momentum = momentum

        model(input_var, **kwargs)
        n += b  
    
    #model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return tf.math.log( x / (1.0-x+eps))
    #return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            #torch.manual_seed(seed)
            tf.random.set_seed(seed)
        # if cuda:
        #     input = input.cuda(non_blocking=True)
        
        output = model(input, **kwargs)
        if regression:
            #preds.append(output.cpu().data.numpy())
            preds.append(output.numpy())
        else:
            # probs = F.softmax(output, dim=1)
            # preds.append(probs.cpu().data.numpy())
            #probs = tf.nn.softmax(output, axis=1)
            probs = output
            preds.append(probs.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor
