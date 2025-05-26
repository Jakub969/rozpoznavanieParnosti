# noinspection PyUnresolvedReferences
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, regularizers

def build_model(input_shape, l2_lambda=0.001):
    # augmentacia, zbytočná keďže dáta si generujem
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
    ])
    """
    #Regresný prístup
    #S pool
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 1 neuron pre binárnu klasifikáciu
    ])
    """
    #bez pool
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    """
    #Klasifikačný prístup
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda),
                      input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(2, activation='softmax')  # Výstup pre 2 triedy: [1,0] = nepárny, [0,1] = párny
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
