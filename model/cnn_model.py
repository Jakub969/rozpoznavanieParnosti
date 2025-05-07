# noinspection PyUnresolvedReferences
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers

def build_model(input_shape):
    # augmentacia
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
    ])

    model = keras.Sequential([
        data_augmentation,  # <- augmentácia priamo v modeli
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 1 neuron pre binárnu klasifikáciu
    ])
    #bez pool
    #plne prepojenu

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
