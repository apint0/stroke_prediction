"""Task-specific metrics used during training and validation stages."""
from __future__ import print_function
from keras import backend as K


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    y_true = y_true[:, :, 1]
    shape = K.shape(y_true)
    y_true_s = K.reshape(y_true, (shape[0], shape[1], 1))

    enc_0 = K.zeros_like(y_true_s)
    enc_1 = K.ones_like(y_true_s)
    encoder = K.concatenate([enc_0, enc_1])

    y_pred = K.sum(y_pred * encoder, axis=-1)

    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    den = K.epsilon() + K.sum(y_pred)

    prec = intersection / den
    # prec = (K.mean(prec))

    return prec


def recall(y_true, y_pred):

    y_true = y_true[:, :, 1]
    shape = K.shape(y_true)
    y_true_s = K.reshape(y_true, (shape[0], shape[1], 1))

    enc_0 = K.zeros_like(y_true_s)
    enc_1 = K.ones_like(y_true_s)
    encoder = K.concatenate([enc_0, enc_1])

    y_pred = K.sum(y_pred * encoder, axis=-1)

    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    den = K.epsilon() + K.sum(y_true)

    rec = intersection / den
    # rec = K.mean(rec)

    return rec


def hard_dice(y_true, y_pred):

    y_true = y_true[:, :, 1]
    shape = K.shape(y_true)
    y_true_s = K.reshape(y_true, (shape[0], shape[1], 1))

    enc_0 = K.zeros_like(y_true_s)
    enc_1 = K.ones_like(y_true_s)
    encoder = K.concatenate([enc_0, enc_1])

    y_pred = K.sum(y_pred * encoder, axis=-1)

    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = 2.0 * K.sum(y_true * y_pred)
    den = K.epsilon() + K.sum(y_true) + K.sum(y_pred)

    dice = intersection / den
    # dice = K.mean(dice)

    return dice


def soft_dice_loss(y_true, y_pred):

    y_true = y_true[:, :, 1]
    shape = K.shape(y_true)
    y_true_s = K.reshape(y_true, (shape[0], shape[1], 1))

    enc_0 = K.zeros_like(y_true_s)
    enc_1 = K.ones_like(y_true_s)
    encoder = K.concatenate([enc_0, enc_1])

    y_pred = K.sum(y_pred * encoder, axis=-1)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = 2.0 * K.sum(y_true * y_pred)
    den = K.epsilon() + K.sum(y_true) + K.sum(y_pred)
    dice = intersection / den
    dice_loss = 1.0 - dice
    # dice = K.mean(dice)

    return dice_loss
