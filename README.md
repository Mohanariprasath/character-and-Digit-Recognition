<h1>EMNIST Balanced — Character Classification using ANN (TensorFlow)</h1>

<p>
This project trains a Fully Connected Neural Network (ANN / MLP) on the 
<strong>EMNIST Balanced</strong> dataset using <strong>TensorFlow</strong> 
and <strong>TensorFlow Datasets (TFDS)</strong>.  
The model learns to classify 47 different characters including digits, letters, and symbols.
</p>

<hr>

<h2>📦 Requirements</h2>

<pre><code>pip install tensorflow tensorflow-datasets matplotlib numpy
</code></pre>

<hr>

<h2>📂 Dataset</h2>

<p>The dataset is loaded using:</p>

<pre><code>ds_train, ds_test, ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
</code></pre>

<ul>
  <li>131,600 training samples</li>
  <li>18,800 test samples</li>
  <li>47 classes (digits + merged letters)</li>
</ul>

<hr>

<h2>🧹 Preprocessing</h2>

<p>Each image is normalized and flattened from <strong>28×28 → 784</strong>:</p>

<pre><code>def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])
    return image, label
</code></pre>

<hr>

<h2>🧠 Model Architecture</h2>

<pre><code>Input (784)
↓
Dense(512, relu) + Dropout(0.3)
↓
Dense(256, relu) + Dropout(0.3)
↓
Dense(47, softmax)
</code></pre>

<p>Compiled using:</p>

<pre><code>model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
</code></pre>

<hr>

<h2>🚀 Training</h2>

<pre><code>history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test
)
</code></pre>

<hr>

<h2>📈 Evaluation</h2>

<pre><code>test_loss, test_acc = model.evaluate(ds_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
</code></pre>

<p><strong>Typical accuracy:</strong> ~86–89%</p>

<hr>

<h2>📊 Predictions</h2>

<p>The script displays 10 sample predictions:</p>

<pre><code>for images, labels in ds_test.take(1):
    preds = model.predict(images)
    preds_cls = np.argmax(preds, axis=1)
</code></pre>

<p>Visualized using:</p>

<pre><code>plt.imshow(tf.reshape(images[i], (28, 28)), cmap='gray')
plt.title(f"True: {labels[i].numpy()} | Pred: {preds_cls[i]}")
</code></pre>

<hr>

<h2>📁 Project Structure</h2>

<pre><code>📦 EMNIST-MLP
 ┣ 📄 README.md
 ┣ 📄 emnist_mlp.py
 ┗ 📄 requirements.txt
</code></pre>

<hr>

<h2>🔮 Improvements</h2>

<ul>
  <li>Use a CNN for better accuracy</li>
  <li>Add TensorBoard logs</li>
  <li>Add learning curves</li>
  <li>Save and load trained models</li>
  <li>Add confusion matrix</li>
</ul>

<hr>

<h2>📝 License</h2>

<p>Open-source under the <strong>MIT License</strong>.</p>
