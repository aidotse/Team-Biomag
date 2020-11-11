import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy


def binary_focal_loss(gamma=2.0, alpha=0.25, sum=False):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)

        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise

        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise

        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        if sum:
            loss = K.sum(loss, axis=1)
        return loss

    return focal_loss


def channelwise_loss(y_true, y_pred):
    total_loss = 0.
    for ch in [1, 2]:
        total_loss += MeanSquaredError()(y_true[..., ch], y_pred[..., ch])

    # The first channel is the nuclei
    # Most of the pixels below the intensity 600 are the part of the background and correlates with the cyto.
    # Therefore we concentrate on the >600 ground truth pixels. (600 ~ .1 after normalization)
    nuclei_weight = .8
    nuclei_thresh = .1

    nuclei_weight_tensor = nuclei_weight * tf.cast(y_true[..., 0] > nuclei_thresh, tf.float32) + (
            1. - tf.cast(y_true[..., 0] <= nuclei_thresh, tf.float32))

    nuclei_loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(
        y_true[..., 0],
        y_pred[..., 0]
    )

    total_loss += tf.math.reduce_mean(nuclei_loss * nuclei_weight_tensor)

    return total_loss

def ssim_loss(max_val=255, alpha=0.84, channel_weights=None, **kwargs):
    if channel_weights is None:
        def ssim(y_true: tf.Tensor, y_pred: tf.Tensor):
            return alpha * (1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val))) + \
                   (1 - alpha) * tf.losses.mae(y_true, y_pred)
    elif channel_weights == "mean":
        def ssim(y_true: tf.Tensor, y_pred: tf.Tensor):
            channel_weights = tf.reduce_mean(y_true, (1, 2))
            channel_weights = channel_weights/tf.reduce_sum(channel_weights)
            loss = 0.
            for c in range(y_true.shape[-1]):
                loss += alpha*(1-tf.image.ssim_multiscale(y_true[..., c], y_pred[..., c], max_val))*channel_weights[c]
                loss += (1-alpha)*tf.losses.mae(y_true, y_pred)
            return loss
    else:
        def ssim(y_true: tf.Tensor, y_pred: tf.Tensor):
            loss = 0.
            for c in range(len(channel_weights)):
                loss += alpha * (1 - tf.image.ssim(y_true[..., c:c+1], y_pred[..., c:c+1], max_val)) * \
                        channel_weights[c]
                loss += (1 - alpha) * tf.losses.mae(y_true, y_pred)
            return loss
    return ssim

def ssim(max_val=255, channel_weights=None, **kwargs):
    def ssim(y_true, y_pred):
        if channel_weights is None:
            return tf.reduce_sum(tf.image.ssim(y_true, y_pred, max_val, **kwargs))

    return ssim