import tensorflow as tf
from tensorflow.keras import backend as K

# Generalized Dice Loss
def generalized_dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    total = K.sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - K.mean((2. * intersection + smooth) / (total + smooth), axis=0)

# Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# Mean Absolute Error (MAE) Loss
def mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred), axis=-1)

# Hinge Loss (for classification)
def hinge_loss(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0), axis=-1)

# Log-Cosh Loss
def log_cosh_loss(y_true, y_pred):
    return K.mean(K.log(K.cosh(y_pred - y_true)), axis=-1)

# Soft Dice Loss (for imbalanced segmentation tasks)
def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    return 1 - K.mean((2. * intersection + smooth) / (K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) + smooth), axis=0)

# Categorical Crossentropy Loss
def categorical_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Binary Cross-Entropy Loss (BCE)
def binary_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Focal Loss for Class Imbalance
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fixed

# Custom Metric: True Positive Rate (Recall)
def true_positive_rate(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positive / (possible_positive + K.epsilon())

# Custom Metric: True Negative Rate (Specificity)
def true_negative_rate(y_true, y_pred):
    true_negative = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negative = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negative / (possible_negative + K.epsilon())

# Matthews Correlation Coefficient (MCC)
def mcc(y_true, y_pred):
    epsilon = K.epsilon()
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    numerator = tp * tn - fp * fn
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + epsilon)
    return numerator / (denominator + epsilon)

# AUC (Area Under Curve) - ROC
def auc_roc_metric(y_true, y_pred):
    return tf.keras.metrics.AUC()(y_true, y_pred)

# Precision-Recall AUC
def auc_pr_metric(y_true, y_pred):
    return tf.keras.metrics.AUC(curve='PR')(y_true, y_pred)

# Combined Loss: MSE + Dice Loss
def mse_dice_loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    dice = generalized_dice_loss(y_true, y_pred)
    return mse + dice

# Combined Loss: BCE + Focal Loss
def bce_focal_loss(y_true, y_pred):
    bce = binary_crossentropy_loss(y_true, y_pred)
    focal = focal_loss()(y_true, y_pred)
    return bce + focal

# Combined Metric: AUC + Dice
def auc_dice_metric(y_true, y_pred):
    auc_metric = auc_roc_metric(y_true, y_pred)
    dice_metric = generalized_dice_loss(y_true, y_pred)
    return auc_metric + dice_metric

# Weighted Cross-Entropy Loss
def weighted_binary_crossentropy(y_true, y_pred, weight_pos=0.5, weight_neg=0.5):
    bce = binary_crossentropy_loss(y_true, y_pred)
    return weight_pos * bce * y_true + weight_neg * bce * (1 - y_true)

# IoU (Intersection over Union)
def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

# Custom Metric: Log-Cosh Error
def log_cosh_error(y_true, y_pred):
    return K.mean(K.log(K.cosh(y_pred - y_true)), axis=-1)

# Custom Metric: Mean Absolute Percentage Error (MAPE)
def mape_metric(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / (y_true + K.epsilon())), axis=-1)

# Custom Metric: R-Squared (R2)
def r_squared(y_true, y_pred):
    ss_total = K.sum(K.square(y_true - K.mean(y_true)))
    ss_residual = K.sum(K.square(y_true - y_pred))
    return 1 - ss_residual / (ss_total + K.epsilon())
