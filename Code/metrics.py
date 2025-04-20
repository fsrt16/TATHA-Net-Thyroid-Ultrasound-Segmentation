import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient loss to evaluate segmentation quality.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Dice loss value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Binary Cross-Entropy + Dice Loss
def bce_dice_loss(y_true, y_pred):
    """
    Combines Binary Cross-Entropy and Dice loss.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :return: Sum of BCE and Dice loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# Tversky Loss
def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Tversky loss function for imbalanced datasets.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param alpha: Weight for false negatives.
    :param beta: Weight for false positives.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Tversky loss value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

# Focal Tversky Loss
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky loss for harder-to-classify examples.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param alpha: Weight for false negatives.
    :param beta: Weight for false positives.
    :param gamma: Focusing parameter for hard examples.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Focal Tversky loss value.
    """
    tversky = tversky_loss(y_true, y_pred, alpha, beta, smooth)
    return K.pow(tversky, gamma)

# Weighted Cross-Entropy Loss
def weighted_bce_loss(y_true, y_pred, pos_weight=1.0, neg_weight=1.0):
    """
    Weighted Binary Cross-Entropy loss to handle class imbalance.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param pos_weight: Weight for positive class.
    :param neg_weight: Weight for negative class.
    :return: Weighted BCE loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(y_true, y_pred)
    return loss * (pos_weight * y_true + neg_weight * (1 - y_true))

# Jaccard Loss
def jaccard_loss(y_true, y_pred, smooth=1e-6):
    """
    Jaccard loss (IoU loss) for segmentation.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Jaccard loss value.
    """
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (intersection + smooth) / (union + smooth)

# Combined Jaccard and Dice Loss
def jac_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Combined Jaccard and Dice loss to improve segmentation.
    :param y_true: Ground truth binary mask.
    :param y_pred: Predicted binary mask.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Sum of Jaccard and Dice loss.
    """
    jac_loss = jaccard_loss(y_true, y_pred, smooth)
    dice_loss_val = dice_loss(y_true, y_pred, smooth)
    return jac_loss + dice_loss_val

# Metrics Class for Evaluation
class Metrics(tf.keras.metrics.Metric):
    def __init__(self, name='metrics', **kwargs):
        super(Metrics, self).__init__(name=name, **kwargs)
        self.dice_coefficient = self.add_weight(name='dice', initializer='zeros')
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.auc_roc = tf.keras.metrics.AUC()
        self.ap_score = tf.keras.metrics.AUC(multi_label=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of each metric during training.
        """
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        # Update each metric
        self.dice_coefficient.assign(self.dice_coefficient_func(y_true, y_pred))
        self.iou.assign(self.iou_func(y_true, y_pred))
        self.precision.update_state(y_true_flat, y_pred_flat, sample_weight)
        self.recall.update_state(y_true_flat, y_pred_flat, sample_weight)
        self.auc_roc.update_state(y_true_flat, y_pred_flat, sample_weight)
        self.ap_score.update_state(y_true_flat, y_pred_flat, sample_weight)

    def result(self):
        """
        Return the computed metrics.
        """
        return {
            'dice_coefficient': self.dice_coefficient,
            'iou': self.iou,
            'precision': self.precision.result(),
            'recall': self.recall.result(),
            'auc_roc': self.auc_roc.result(),
            'average_precision': self.ap_score.result(),
        }

    def reset_states(self):
        """
        Reset the metrics states.
        """
        super(Metrics, self).reset_states()
        self.dice_coefficient.assign(0)
        self.iou.assign(0)

    def dice_coefficient_func(self, y_true, y_pred, smooth=1e-6):
        """
        Compute the Dice coefficient.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def iou_func(self, y_true, y_pred):
        """
        Compute the Intersection over Union (IoU).
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return intersection / (union + K.epsilon())

# ROC AUC and Average Precision Score Metrics
def roc_auc_score_metric(y_true, y_pred):
    """
    Compute ROC AUC score for binary classification.
    """
    return roc_auc_score(y_true, y_pred)

def average_precision_metric(y_true, y_pred):
    """
    Compute Average Precision Score.
    """
    return average_precision_score(y_true, y_pred)

# Root Mean Squared Error (RMSE) Loss
def rmse_loss(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) loss function.
    :param y_true: Ground truth.
    :param y_pred: Predicted values.
    :return: RMSE loss value.
    """
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# Hinge Loss
def hinge_loss(y_true, y_pred):
    """
    Hinge loss function for classification problems.
    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :return: Hinge loss value.
    """
    return K.mean(K.maximum(1. - y_true * y_pred, 0.))

# Mean Absolute Error (MAE) Loss
def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error loss function.
    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :return: MAE loss value.
    """
    return K.mean(K.abs(y_true - y_pred))
