# Import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

def verify_setup():
    """Verify TensorFlow setup and GPU availability"""
    print("TensorFlow version:", tf.__version__)
    print("Eager execution:", tf.executing_eagerly())
    print("Hub version:", hub.__version__)
    print("GPU is", "available" if len(tf.config.list_physical_devices('GPU')) > 0 else "NOT AVAILABLE")

def setup_data_directories(zip_url):
    """Download and setup data directories"""
    zip_file = tf.keras.utils.get_file(
        origin=zip_url,
        fname='rice-leaf.zip',
        extract=True
    )
    
    # Setup directories
    data_dir = os.path.join(os.path.dirname(zip_file), 'rice-leaf_extracted')
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    
    return data_dir, train_dir, validation_dir

def load_classes(json_path):
    """Load class labels from JSON file"""
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    return cat_to_name, classes

def setup_model_config():
    """Configure model parameters"""
    module_selection = ("inception_v3", 299, 2048)
    handle_base, pixels, FV_SIZE = module_selection
    MODULE_HANDLE = f"https://tfhub.dev/google/tf2-preview/{handle_base}/feature_vector/4"
    IMAGE_SIZE = (pixels, pixels)
    BATCH_SIZE = 32
    return MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, BATCH_SIZE

def create_data_generators(train_dir, validation_dir, IMAGE_SIZE, BATCH_SIZE):
    """Create data generators for training and validation"""
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, 
        shuffle=False, 
        seed=42,
        color_mode="rgb", 
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        shear_range=0.2, 
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir, 
        shuffle=True, 
        seed=42,
        color_mode="rgb", 
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, validation_generator

def build_model(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, num_classes, trainable=False):
    """Build and compile the model using Functional API with TF Hub"""

    # Input layer
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Wrap hub.KerasLayer in Lambda to avoid symbolic tensor issues
    x = tf.keras.layers.Lambda(lambda img: hub.KerasLayer(
        MODULE_HANDLE, trainable=trainable, dtype=tf.float32)(img)
    )(inputs)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, train_generator, validation_generator, epochs=30):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    return history

def plot_training_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(filename, IMAGE_SIZE):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def predict_image(model, image, classes):
    img_array = np.expand_dims(image, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return {classes[class_idx]: confidence}

def predict_folder(folder_path, model_path='rice_leaf_disease_model.h5', json_path='classes.json', IMAGE_SIZE=(299, 299)):
    """Predict all images in a folder using the trained model and a classes.json file."""
    import tensorflow_hub as hub
    import tensorflow as tf
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    # Load class names from JSON
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    # Build the model
    MODULE_HANDLE, _, FV_SIZE, _ = setup_model_config()
    num_classes = len(cat_to_name)
    model = build_model(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, num_classes, trainable=False)

    # Load weights
    try:
        model.load_weights(model_path)
    except Exception:
        model.load_weights(model_path, by_name=True)

    # Ensure classes list matches model output order
    classes = [cat_to_name[key] for key in sorted(cat_to_name.keys())]

    # List all images in folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = load_and_preprocess_image(img_path, IMAGE_SIZE)
        prediction = predict_image(model, img, classes)

        # Display image with prediction
        plt.imshow(img)
        plt.title(f"Image: {img_file}\nPredicted: {list(prediction.keys())[0]}\nConfidence: {list(prediction.values())[0]*100:.2f}%")
        plt.axis('off')
        plt.show()

        print(f"\nImage: {img_file}")
        print("Prediction:", prediction)
        print("-"*40)

def main():
    verify_setup()
    zip_url = 'https://github.com/AveyBD/rice-leaf-diseases-detection/raw/master/rice-leaf.zip'
    data_dir, train_dir, validation_dir = setup_data_directories(zip_url)
    cat_to_name, classes = load_classes('classes.json')
    print(f"Number of classes: {len(classes)}")
    MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, BATCH_SIZE = setup_model_config()
    train_generator, validation_generator = create_data_generators(
        train_dir, validation_dir, IMAGE_SIZE, BATCH_SIZE
    )
    model = build_model(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, len(classes), trainable=False)
    model.summary()
    history = train_model(model, train_generator, validation_generator, epochs=10)
    plot_training_history(history, epochs=10)
    model.save('rice_leaf_disease_model.h5')
    print("✅ Model saved successfully as 'rice_leaf_disease_model.h5'")

if __name__ == "__main__":
    # Commented out training for prediction testing
    #main()
    
    # Predict images from folder
    predict_folder("test_images", model_path='rice_leaf_disease_model.h5', json_path='classes.json', IMAGE_SIZE=(299, 299))