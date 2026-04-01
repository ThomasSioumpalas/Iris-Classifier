Iris Classifier

A neural network classifier trained on the Iris flower dataset using TensorFlow 2.x and Keras.

## What it does

- Loads the Iris dataset from TensorFlow Datasets (TFDS)
- Normalizes the 4 flower measurements (sepal/petal length & width)
- Trains a 3-layer neural network to classify flowers into 3 species
- Evaluates accuracy on a held-out test set
- Makes a prediction on a single new flower sample

## Requirements

pip install tensorflow tensorflow-datasets

## Output

- Test loss and accuracy printed after training
- Predicted species for the sample flower (setosa / versicolor / virginica)

## Dataset

150 iris flower samples, 4 numerical features each, 3 classes.
Split: 80% train / 20% test.
