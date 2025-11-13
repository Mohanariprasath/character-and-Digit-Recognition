import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])
    return image, label

batch_size = 128
ds_train = ds_train.map(normalize_img).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

num_classes = ds_info.features['label'].num_classes

model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(ds_train, epochs=10, validation_data=ds_test)

test_loss, test_acc = model.evaluate(ds_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

import numpy as np

for images, labels in ds_test.take(1):
    preds = model.predict(images)
    preds_cls = np.argmax(preds, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(tf.reshape(images[i], (28, 28)), cmap='gray')
        plt.title(f"True: {labels[i].numpy()} | Pred: {preds_cls[i]}")
        plt.axis('off')
    plt.show()
