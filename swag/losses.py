import torch
import torch.nn.functional as F

import tensorflow as tf 
import tensorflow_probability as tfp
tfd = tfp.distributions

def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    #loss = F.cross_entropy(output, target)
    #print(output[0])
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    #loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_fn(target, output)
    #loss = tf.math.reduce_sum(loss)


    return loss, output


def adversarial_cross_entropy(
    model, input, target, lossfn=tf.keras.losses.SparseCategoricalCrossentropy(), epsilon=0.01
):
    # loss function based on algorithm 1 of "simple and scalable uncertainty estimation using
    # deep ensembles," lakshminaraynan, pritzel, and blundell, nips 2017,
    # https://arxiv.org/pdf/1612.01474.pdf
    # note: the small difference bw this paper is that here the loss is only backpropped
    # through the adversarial loss rather than both due to memory constraints on preresnets
    # we can change back if we want to restrict ourselves to VGG-like networks (where it's fine).

    # scale epsilon by min and max (should be [0,1] for all experiments)
    # see algorithm 1 of paper
    scaled_epsilon = epsilon * (tf.keras.backend.max(input) - tf.keras.backend.min(input))

    # force inputs to require gradient
    #input.requires_grad = True

    # standard forwards pass
    # output = model(input)
    # loss = lossfn(target, output)
    #loss = lossfn(output, target)

    # now compute gradients wrt input
    with tf.GradientTape() as tape:
        input = tf.Variable(input)
        output = model(input)
        tape.watch(input)
        loss = lossfn(target, output)
        inputs_grad = tape.gradient(loss, input)


    #loss.backward(retain_graph=True)

    # now compute sign of gradients
    #inputs_grad = torch.sign(input.grad)
    inputs_grad = tf.math.sign(inputs_grad)

    # perturb inputs and use clamped output
    # inputs_perturbed = torch.clamp(
    #     input + scaled_epsilon * inputs_grad, 0.0, 1.0
    # ).detach()
    inputs_perturbed = tf.clip_by_value(
        input + scaled_epsilon * inputs_grad, 0.0, 1.0
    )
    # inputs_perturbed.requires_grad = False

    #input.grad.zero_()
    # model.zero_grad()

    outputs_perturbed = model(inputs_perturbed)

    # compute adversarial version of loss
    #adv_loss = lossfn(outputs_perturbed, target)
    adv_loss = lossfn(target, outputs_perturbed)

    # return mean of loss for reasonable scalings
    return (loss + adv_loss) / 2.0, output


def masked_loss(y_pred, y_true, void_class=11.0, weight=None, reduce=True):
    # masked version of crossentropy loss

    # el = torch.ones_like(y_true) * void_class
    # mask = torch.ne(y_true, el).long()

    el = tf.ones_like(y_true) * void_class
    mask = tf.math.not_equal(y_true, el)

    y_true_tmp = y_true * mask

    # loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction="none")
    # loss = mask.float() * loss

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = loss_fn(y_true_tmp, y_pred, sample_weight=weight)
    loss = tf.cast(mask, tf.float32) * loss

    if reduce:
        #return loss.sum() / mask.sum()
        return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)
    else:
        return loss, mask


def seg_cross_entropy(model, input, target, weight=None):
    output = model(input)

    # use masked loss function
    loss = masked_loss(output, target, weight=weight)

    return {"loss": loss, "output": output}


def seg_ale_cross_entropy(model, input, target, num_samples=50, weight=None):
    # requires two outputs for model(input)

    output = model(input)
    mean = output[:, 0, :, :, :]
    scale = tf.math.abs(output[:, 1, :, :, :])

    #output_distribution = torch.distributions.Normal(mean, scale)
    output_distribution = tfd.Normal(loc=mean, scale=scale)


    total_loss = 0

    for _ in range(num_samples):
        #sample = output_distribution.rsample()
        sample = output_distribution.sample()
        current_loss, mask = masked_loss(sample, target, weight=weight, reduce=False)
        total_loss = total_loss + tf.math.exp(current_loss)
        #total_loss = total_loss + current_loss.exp()
    mean_loss = total_loss / num_samples
    return {"loss": tf.math.reduce_sum(tf.math.log(mean_loss))/tf.math.reduce_sum(mask),
            "output": mean, "scale": scale
         }
    # return {"loss": mean_loss.log().sum() / mask.sum(), "output": mean, "scale": scale}
