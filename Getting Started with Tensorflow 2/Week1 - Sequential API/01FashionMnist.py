import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D

#Build a sequential feedforward neural network model

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(16, activation='relu'),
    Dense(16,activation='relu'),
    Dense(10, activation='softmax'),
])

#Print the model summary
model.summary()

#Build a convolutional neural network model
model = Sequential([
    Conv2D(16,(3,3),activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(3,3),
    Flatten(),
    Dense(10, activation='softmax')
])

#Print the model summary
model.summary()

#Define the model optimizer, loss function and metrics
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt,
              loss = 'sparse_categorical_crossentropy',
              metrics = [acc,mae])

#Print the resulting model attributes
print(model.loss)
print(model.optimizer)
print(model.metrics)
print(model.optimizer.lr)

from tensorflow.keras.preprocessing import  image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Load the fashion Mnist dataset
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

#Print the shape of training and test data
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

#Define the labels
labels = ['T-shirt/top',
          'Trouser',
          'Pullover',
          'Dress',
          'Coat',
          'Sandal',
          'Shirt',
          'Sneaker',
          'Bag',
          'Ankle Boot']

print(train_labels[0])

#Rescale the image values so that they lie between zero and one.
train_images = train_images/255.
test_images = test_images/255.

#Display one of the images
i=0
img = train_images[i,:,:]
plt.imshow(img)
plt.show()
print(f"label: {labels[train_labels[i]]}")

#fit the model
history = model.fit(train_images[...,np.newaxis], train_labels, epochs=8,
                    batch_size=256)

#Load the history into a pandas DataFrame
df = pd.DataFrame(history.history)
df.head()

#Make a plot for the loss
loss_plot = df.plot(y="loss",
                    title="Loss vs Epochs",
                    legend = False)
loss_plot.set(xlabel="Epochs", ylabel="Loss")

#Make a plot for the accuracy
loss_plot=df.plot(y="sparse_categorical_accuracy",
                  title = "Accuracy Vs Epochs",
                  legend = False)
loss_plot.set(xlabel = "Epochs", ylabel="Accuracy")

#Make a plot for the additional Metric
loss_plot = df.plot(y="mean_absolute_error",
                    title = "MAE vs Epochs",
                    legend = False)
loss_plot.set(xlabel = "Epochs", ylabel = "MAE")

test_loss, test_accuracy, test_mae = model.evaluate(test_images[...,np.newaxis],
                                                    test_labels, verbose=2)

#Choose a random test image
random_inx = np.random.choice(test_images.shape[0])

test_image = test_images[random_inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[random_inx]]}")

predictions = model.predict(test_image[np.newaxis,...,np.newaxis])
print(f"Model Prediction: {labels[np.argmax(predictions)]}")


