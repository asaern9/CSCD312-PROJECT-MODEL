from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import  DenseNet201

#building model
classes = 2

def build_model():
    model = Sequential(
        [
            DenseNet201(weights='imagenet',include_top=False,input_shape=(224,224,3)),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(classes, activation='softmax'),
        ]
    )
    
    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=1e-4),metrics=['accuracy'])
    return model

model = build_model()
model.summary()