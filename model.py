import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (128, 128)
MODEL_FILENAME = "brain_tumor_multiclass_cnn.h5"

def build_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Builds a Multiclass CNN architecture for Brain Tumor Classification
    corresponding to classes: glioma, meningioma, notumor, pituitary.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def generate_model(save_path=MODEL_FILENAME, epochs=5, batch_size=32):
    """
    Trains the CNN model if dataset folders exist; otherwise, generates and saves
    the initialized brain tumor multiclass CNN model ready for inference.
    """
    model = build_model()
    
    train_dir = "training"
    test_dir = "testing"
    
    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        print(f"Dataset directory '{train_dir}' found. Training model...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode='categorical'
        )
        val_ds = None
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            val_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=IMG_SIZE,
                batch_size=batch_size,
                label_mode='categorical'
            )
        
        # Normalize datasets
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        if val_ds:
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
            
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    else:
        print(f"No local training dataset folder found at '{train_dir}'. Initializing and saving baseline CNN model...")

    model.save(save_path)
    print(f"Model saved successfully to {save_path}")
    return model

if __name__ == "__main__":
    generate_model()
