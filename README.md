

  <div class="hero">
    <h1>EMNIST Balanced — Character Classification using ANN (TensorFlow)</h1>
    <p>
      A simple fully-connected neural network (MLP) example that trains on the
      <strong>EMNIST Balanced</strong> dataset using <strong>TensorFlow</strong> and <strong>tensorflow-datasets</strong>.
    </p>
    <div>
      <span class="badge">Dataset: EMNIST Balanced</span>
      <span class="badge">Model: MLP (Dense)</span>
    </div>
  </div>


  <h2>Overview</h2>
  <p>This repository contains a straightforward script that:</p>
  <ul>
    <li>loads <code>emnist/balanced</code> from TFDS</li>
    <li>normalizes and flattens 28×28 images → 784 features</li>
    <li>trains an MLP: Dense(512) → Dropout → Dense(256) → Dropout → Dense(num_classes)</li>
    <li>evaluates on the test split and visualizes sample predictions</li>
  </ul>

  <h2>Rendered sample output</h2>
  <p>The image below is embedded directly (base64 PNG). It typically shows sample predictions produced after training.</p>

  <figure>
   <img width="950" height="454" alt="output" src="https://github.com/user-attachments/assets/ede7523b-44b9-4a8d-86d8-47e8a90b7677" />
    <figcaption>Sample predictions and training output (embedded PNG)</figcaption>
  </figure>

  <h2>Quick start</h2>
  <ol>
    <li>Create a virtual environment and install dependencies:
      <pre><code>pip install tensorflow tensorflow-datasets matplotlib numpy</code></pre>
    </li>
    <li>Run the training script (example):
      <pre><code>python emnist_mlp.py --epochs 10 --batch_size 128</code></pre>
    </li>
    <li>View the printed evaluation result and sample prediction figure (the image above is one example output).</li>
  </ol>

  <h2>Preprocessing snippet</h2>
  <pre><code>def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (-1,))  # flatten to 784
    return image, label
  </code></pre>

  <h2>Model snippet</h2>
  <pre><code>model = Sequential([
  Input(shape=(28*28,)),
  Dense(512, activation='relu'),
  Dropout(0.3),
  Dense(256, activation='relu'),
  Dropout(0.3),
  Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])</code></pre>

  <h2>Notes & improvements</h2>
  <ul>
    <li>For better accuracy, replace the MLP with a small CNN (Conv2D→Pool→Dense).</li>
    <li>EMNIST sometimes requires rotating/transposing images depending on split — verify orientation before using in production.</li>
    <li>Add model checkpointing: <code>model.save(...)</code> or <code>tf.keras.callbacks.ModelCheckpoint</code>.</li>
  </ul>


