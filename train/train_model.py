import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from dataset.split_dataset import DatasetSplitter

def train_and_evaluate(model, dataset_path, data_subset="all"):
    # Načítanie dát podľa výberu
    if data_subset in ["noise", "shapes"]:
        splitter = DatasetSplitter(dataset_path)
        noise_data, shape_data = splitter.split_by_type()

        if data_subset == "noise":
            images = noise_data['images']
            labels = noise_data['labels']
        elif data_subset == "shapes":
            images = shape_data['images']
            labels = shape_data['labels']
    else:
        images = np.load(f"{dataset_path}/images.npy")
        labels = np.load(f"{dataset_path}/labels.npy")

    # Prevod labelov na one-hot vektor [1, 0] = nepárny, [0, 1] = párny
    labels = to_categorical(labels, num_classes=2)

    # Predspracovanie
    images = images.reshape((-1, 28, 28, 1)).astype('float32')

    # Rozdelenie dát
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # Tréning modelu
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # Vyhodnotenie modelu
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Vizualizácia priebehu tréningu
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

    # Uloženie modelu
    #model.save(f"parnost_model_{data_subset}.h5")
