import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16

# -------------------------------------------------------------------
# DATA LOADING + AUGMENTATION
# This block handles image preprocessing. We use ImageDataGenerator
# to automatically label images based on folder names and apply
# light augmentation (rotation, zoom, flip). This improves model
# generalization without manually editing images ourselves.
# -------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = len(train_data.class_indices)
print("Classes:", train_data.class_indices)

# -------------------------------------------------------------------
# TRANSFER LEARNING SETUP
# Instead of training from scratch, we load MobileNetV2, a pretrained
# CNN trained on ImageNet. We freeze its weights so it works as a
# “feature extractor,” and then place our own classifier layers on
# top. This dramatically reduces training time and improves accuracy
# for small datasets (like our plant mood images).
# -------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze all convolutional layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),     # compress CNN features
    layers.Dense(128, activation="relu"), # small classifier head
    layers.Dropout(0.3),                 # prevent overfitting
    layers.Dense(num_classes, activation="softmax") # prediction output
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------------------------
# MODEL TRAINING
# We train the classifier head for 10 epochs. The pretrained base
# stays frozen. The model learns to map CNN features to our plant
# “mood” categories. Validation data helps track generalization.
# -------------------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("plant_mood_model.h5")

# -------------------------------------------------------------------
# PREDICTION FUNCTION
# This helper function loads a single image, preprocesses it the same
# way as the training data, runs it through our trained model, and
# returns the predicted mood label + raw probabilities. Used for 
# testing new leaf images in real-world scenarios.
# -------------------------------------------------------------------
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_leaf(path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    class_names = list(train_data.class_indices.keys())
    return class_names[np.argmax(preds)], preds

# Example
label, probs = predict_leaf("test_leaf.jpg")
print("Predicted Mood:", label)
