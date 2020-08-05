#Package Imports

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

mnist_data = tf.keras.datasets.mnist
(train_images, train_labels),(test_images,test_labels)=mnist_data.load_data()

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

def scale_mnist_data(train_images, test_images):
    train_images = train_images/255.
    test_images = test_images/255.
    return (train_images,test_images)

scaled_train_images, scaled_test_images = scale_mnist_data(train_images,test_images)

#Add a dummy channel dimension
scaled_train_images = scaled_train_images[...,np.newaxis]
scaled_test_images = scaled_test_images[...,np.newaxis]

print(scaled_train_images.shape)
print(scaled_test_images.shape)
print(scaled_train_images[0].shape)

# Build the convolutional neural network model
# We are now ready to construct a model to fit to the data. Using the Sequential API, build your CNN model according to the following spec:
#
# The model should use the input_shape in the function argument to set the input size in the first layer.
# A 2D convolutional layer with a 3x3 kernel and 8 filters. Use 'SAME' zero padding and ReLU activation functions. Make sure to provide the input_shape keyword argument in this first layer.
# A max pooling layer, with a 2x2 window, and default strides.
# A flatten layer, which unrolls the input into a one-dimensional tensor.
# Two dense hidden layers, each with 64 units and ReLU activation functions.
# A dense output layer with 10 units and the softmax activation function.
# In particular, your neural network should have six layers.

def get_model(input_shape):
    model = Sequential([
        Conv2D(8,(3,3),padding="SAME", activation='relu',
               input_shape=(input_shape)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model
model = get_model(scaled_train_images[0].shape)
model.summary()

def compile_model(model):
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

compile_model(model)

print(model.optimizer)
print(model.loss)
print(model.metrics)

def train_model(model, scaled_train_images, train_labels):
    history = model.fit(scaled_train_images, train_labels, epochs=5, verbose=2)
    return history

history = train_model(model, scaled_train_images, train_labels)

frame = pd.DataFrame(history.history)

acc_plot = frame.plot(y="accuracy",
                      title = "Accuracy Vs Epochs",
                      legend = False)
acc_plot.set(xlabel = "Epochs", ylabel = "Accuracy")

acc_plot = frame.plot(y="loss", title = "Loss vs Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

def evaluate_model(model, scaled_test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(scaled_test_images,test_labels, verbose=2)
    return (test_loss, test_accuracy)

test_loss, test_accuracy = evaluate_model(model, scaled_test_images,test_labels)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

num_test_images = scaled_test_images.shape[0]
random_inx = np.random.choice(num_test_images, 4)
random_test_images = scaled_test_images[random_inx,...]
random_test_labels = test_labels[random_inx,...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(4,2,figsize=(16,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images,
                                                   random_test_labels)):
    axes[i,0].imshow(np.squeeze(image))
    axes[i,0].get_xaxis().set_visible(False)
    axes[i,0].get_yaxis().set_visible(False)
    axes[i,0].text(10., -1.5, f'Digit {label}')
    axes[i,1].bar(np.arange(len(prediction)), prediction)
    axes[i,1].set_xticks(np.arange(len(prediction)))
    axes[i,1].set_title(f"Categorical distribution. Model Prediction:{np.argmax(prediction)}")

plt.show()


