#Package imports
from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

def read_in_and_split_data(iris_data):
    iris_data = datasets.load_iris()
    data = iris_data['data']
    targets = iris_data['target']
    train_data, test_data, train_targets, test_targets = train_test_split(
        data, targets, test_size=0.1
    )
    return (train_data,test_data,train_targets,test_targets)

iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(
    iris_data
)
print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)

#Convert targets to a one hot encoding
train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))

def get_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_uniform',
              bias_initializer='ones', input_shape=(input_shape)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

model = get_model(train_data[0].shape)
model.summary()

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics = ['accuracy']
    )

compile_model(model)
print(model.optimizer)
print(model.loss)
print(model.metrics)
print(model.optimizer.lr)

def train_model(model, train_data, train_targets, epochs):
    history = model.fit(
        train_data,
        train_targets,
        epochs=epochs,
        batch_size = 40,
        validation_split=0.15,
        verbose=False
    )
    return history

history = train_model(model, train_data,train_targets, epochs=800)

try:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
except KeyError:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
plt.title("Accuracy vs Epochs")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss vs Epochs")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

def get_regularised_model(input_shape,dropout_rate,weight_decay):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer='he_uniform', bias_initializer='ones',input_shape=(input_shape)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(3, activation='softmax')
    ])
    return model

reg_model = get_regularised_model(train_data[0].shape, 0.3, 0.001)

compile_model(reg_model)

reg_history = train_model(reg_model,train_data,train_targets,epochs=800)

try:
    plt.plot(reg_history.history['accuracy'])
    plt.plot(reg_history.history['val_accuracy'])
except KeyError:
    plt.plot(reg_history.history['acc'])
    plt.plot(reg_history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

def get_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        min_delta=0.01,
        mode='min'
    )
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=20
    )
    return (early_stopping,learning_rate_reduction)

call_model = get_regularised_model(train_data[0].shape, 0.3, 0.0001)
compile_model(call_model)
early_stopping, learning_rate_reduction = get_callbacks()
call_history = call_model.fit(train_data,train_targets,epochs=800,
                              validation_split=0.15,
                              callbacks=[early_stopping,learning_rate_reduction],
                              verbose=0)

print(learning_rate_reduction.patience)

try:
    plt.plot(call_history.history['accuracy'])
    plt.plot(call_history.history['val_accuracy'])
except KeyError:
    plt.plot(call_history.history['acc'])
    plt.plot(call_history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.plot(call_history.history['loss'])
plt.plot(call_history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

test_loss, test_acc = call_model.evaluate(test_data, test_targets, verbose=0)
print("Test Loss: {:.3f}\nTest Accuracy: {:.2f}%".format(test_loss, 100*test_acc))
