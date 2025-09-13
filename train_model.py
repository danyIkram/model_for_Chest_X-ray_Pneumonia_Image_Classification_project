from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sklearn.utils
import numpy as np
import os

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# data augmentation and pretreatment
train_datagen = ImageDataGenerator(
    rescale=1./127.5 - 1,   # Normalisation entre -1 et 1
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./127.5 - 1
)

train_generator = train_datagen.flow_from_directory(
    'dataset/chest_xray/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    'dataset/chest_xray/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# manage unbalanced classes
# Classes : Normal = 0, Pneumonia = 1
y_train = train_generator.classes
weights = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}
print("Class weights:", class_weights)


# create CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes : Normal / Pneumonia
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train with class_weight

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weights
)


# save model and labels

os.makedirs('model', exist_ok=True)
model.save('models/pneumonia_model.h5')

with open('models/labels.txt', 'w') as f:
    for cls, idx in train_generator.class_indices.items():
        f.write(f"{idx} {cls}\n")

