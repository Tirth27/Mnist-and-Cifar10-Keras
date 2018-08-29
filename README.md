# Mnist-and-Cifar10-Keras

Under Development.

## Mnist

The MNIST database of handwritten digits. It has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

The Mnist Classification is a most common "hello world" example for deep learning. It's a dataset of hand-written digits, 0 through 9. It's 28x28 images of these hand-written digits.

###### Dependency
```
pip install --upgrade tensorflow
pip install Keras
```
You can directly load MNIST data using keras.
```
(x_train, y_train),(x_test, y_test) = mnist.load_data()
```
It wil split data into testing and training set. *x_train* contain images  and *y_train* contain labels for the images

We can easy train model using [Keras](https://keras.io/). Keras is a wrapper around tensorflow and uses tensorflow as backend by default.
