import tensorflow as tf
import tensorflow_datasets as tfds

assert tf.__version__.startswith("2.")

# Load TensorFlow Dataset
(ds_train, ds_test), ds_info = tfds.load(
    'iris',
    split=['train[:80%]', 'train[80%:]'],  
    as_supervised=True,
    with_info=True
)

# Peek at one raw sample before any processing
for features, label in ds_train.take(1):
    print("Features:", features.numpy())
    print("Label:", label.numpy())

# preprocess() also normalizes features, and includes shuffle on train
def normalize_data(features, label):
    mean = tf.constant([5.84, 3.05, 3.76, 1.20], dtype=tf.float32)
    std  = tf.constant([0.83, 0.43, 1.76, 0.76], dtype=tf.float32)
    features = (features - mean) / std  # scale all features to same range
    label = tf.one_hot(label, depth=3)  # converting integer label to [0,1,0] style vector
    return features, label

train_data = ds_train.map(normalize_data).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_data  = ds_test.map(normalize_data).batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),  # input_Shape=(4) → input_shape=(4,)
    tf.keras.layers.Dense(64,  activation='relu'),
    tf.keras.layers.Dense(3,   activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, epochs=50, validation_data=test_data)

# Evaluating on preprocessed test_data
loss, accuracy = model.evaluate(test_data)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
