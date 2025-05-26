from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_fcnn_model(input_shape=(28, 28, 1), l2_lambda=0.001):
    #Regresný prístup
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    """
    #Klasifikačný prístup
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(2, activation='softmax')  # 2 triedy: [1,0] = nepárny, [0,1] = párny
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
