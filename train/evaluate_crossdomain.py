import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from dataset.split_dataset import DatasetSplitter  # uprav podľa cesty

def evaluate_shapes_model_on_noise(model_path="parnost_model_shapes.h5", dataset_path="dataset_shapes"):
    # 1. Načítanie modelu
    model = keras.models.load_model(model_path)

    # 2. Načítanie NOISE dát
    splitter = DatasetSplitter(dataset_path)
    noise_data, _ = splitter.split_by_type()

    X_noise = noise_data['images'].reshape((-1, 28, 28, 1)).astype('float32')
    y_noise = to_categorical(noise_data['labels'], num_classes=2)

    # 3. Vyhodnotenie
    loss, accuracy = model.evaluate(X_noise, y_noise, verbose=0)

    print(f"Výkon modelu naučeného na SHAPES pri testovaní na NOISE:")
    print(f"   ➤ Loss: {loss:.4f}")
    print(f"   ➤ Accuracy: {accuracy:.4f}")