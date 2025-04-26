import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_and_evaluate(model, dataset_path):
    # Load data
    images = np.load(f"{dataset_path}/images.npy")
    labels = np.load(f"{dataset_path}/labels.npy")

    images = images.reshape((-1, 28, 28, 1)).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Training
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, validation_data=(X_test, y_test))

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Plotting loss and accuracy
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Tréningová loss')
    plt.plot(history.history['val_loss'], label='Validačná loss')
    plt.title('Strata (Loss) počas tréningu')
    plt.xlabel('Epochy')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Tréningová presnosť')
    plt.plot(history.history['val_accuracy'], label='Validačná presnosť')
    plt.title('Presnosť (Accuracy) počas tréningu')
    plt.xlabel('Epochy')
    plt.ylabel('Presnosť')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save model
    model.save("parnost_model.h5")
