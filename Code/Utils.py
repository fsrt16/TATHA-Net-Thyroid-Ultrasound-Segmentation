import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras import backend as K

# Set seed for reproducibility
def set_seed(seed=42):
    """
    Set random seed for reproducibility across TensorFlow, NumPy, and Python.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to {seed}")

# Image normalization function
def normalize_image(image):
    """
    Normalize images to the range [0, 1].
    """
    return image / 255.0

# Custom Learning Rate Scheduler
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Custom learning rate scheduler that decays the learning rate after each epoch.
    """
    def __init__(self, initial_lr=1e-3, decay_rate=0.96, decay_steps=100):
        super(CustomLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def on_epoch_end(self, epoch, logs=None):
        lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_steps))
        K.set_value(self.model.optimizer.lr, lr)
        print(f"Learning rate updated to: {lr:.6f}")

# Custom Early Stopping
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom early stopping based on validation loss with a patience parameter.
    """
    def __init__(self, patience=5, min_delta=0.01):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.model.stop_training = True

# Save model weights to a specified path
def save_model_weights(model, path):
    """
    Save the model's weights to a specified path.
    """
    model.save_weights(path)
    print(f"Model weights saved to {path}")

# Load model weights from a specified path
def load_model_weights(model, path):
    """
    Load the model's weights from a specified path.
    """
    model.load_weights(path)
    print(f"Model weights loaded from {path}")

# Train-validation-test split
def train_val_test_split(data, labels, val_size=0.2, test_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=val_size + test_size, random_state=random_state)
    val_size_adjusted = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Confusion Matrix Plotting
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix given true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_true, y_pred):
    """
    Plot the ROC curve for binary classification.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# Plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred):
    """
    Plot Precision-Recall curve for binary classification.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.show()

# Learning Rate Finder
def lr_finder(model, train_data, train_labels, batch_size=32, start_lr=1e-7, end_lr=1, num_steps=100):
    """
    Implement Learning Rate Finder method to determine the optimal learning rate.
    """
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_steps)
    losses = []
    
    for lr in lrs:
        K.set_value(model.optimizer.lr, lr)
        history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=1, verbose=0)
        losses.append(history.history['loss'][0])
    
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

# Model Checkpoint
def save_best_model(model, val_loss, best_val_loss, model_path):
    """
    Save the model weights if the validation loss improves.
    """
    if val_loss < best_val_loss:
        model.save_weights(model_path)
        print(f"Best model saved at {model_path}")
        return val_loss
    return best_val_loss

# Adam optimizer with weight decay (L2 regularization)
def adam_optimizer_with_weight_decay(learning_rate=0.001, weight_decay=0.0001):
    """
    Adam optimizer with L2 weight decay.
    """
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)

# Data augmentation for images
def augment_images(images, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1):
    """
    Augment images using various transformations like rotation, zoom, etc.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=True,
        vertical_flip=True
    )
    
    return datagen.flow(images)

# Feature scaling: Normalize dataset to range [0, 1]
def scale_features(X_train, X_test):
    """
    Scale features to the range [0, 1].
    """
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train_scaled = (X_train - min_val) / (max_val - min_val)
    X_test_scaled = (X_test - min_val) / (max_val - min_val)
    
    return X_train_scaled, X_test_scaled

# Label encoding
def label_encoding(y):
    """
    One-hot encode labels for multi-class classification.
    """
    return tf.keras.utils.to_categorical(y)

# Function to display sample images
def display_images(images, labels, n=5):
    """
    Display a few sample images with their corresponding labels.
    """
    plt.figure(figsize=(12, 12))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print accuracy, AUC, etc.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
# Data Generator Class for custom batch processing
class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator class for large datasets that cannot fit in memory.
    """
    def __init__(self, data, labels, batch_size=32, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.data[batch_indexes]
        batch_labels = self.labels[batch_indexes]
        
        return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Random Data Generator (for testing purposes)
def generate_random_data(num_samples=1000, image_size=(64, 64, 3), num_classes=10):
    """
    Generate random data and labels for testing purposes.
    """
    data = np.random.rand(num_samples, *image_size)
    labels = np.random.randint(0, num_classes, num_samples)
    labels = label_encoding(labels)
    return data, labels

# Model Summary Print
def print_model_summary(model):
    """
    Print the summary of the model.
    """
    model.summary()

# Save the model architecture to a JSON file
def save_model_json(model, json_file_path):
    """
    Save model architecture to a JSON file.
    """
    model_json = model.to_json()
    with open(json_file_path, 'w') as json_file:
        json_file.write(model_json)
    print(f"Model architecture saved to {json_file_path}")

# Load the model architecture from a JSON file
def load_model_json(json_file_path):
    """
    Load model architecture from a JSON file.
    """
    from tensorflow.keras.models import model_from_json
    with open(json_file_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    print(f"Model architecture loaded from {json_file_path}")
    return model

# Visualize model layers
def plot_model_layers(model):
    """
    Plot the layers of the model in a hierarchical fashion.
    """
    plot_model(model, to_file='model_layers.png', show_shapes=True, show_layer_names=True)
    img = plt.imread('model_layers.png')
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

