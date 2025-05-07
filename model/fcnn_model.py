from tensorflow import keras
from tensorflow.keras import layers

def build_fcnn_model(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binárna klasifikácia
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
