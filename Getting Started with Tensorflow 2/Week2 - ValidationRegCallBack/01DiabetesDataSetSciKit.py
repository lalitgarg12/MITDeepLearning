import tensorflow as tf
import sklearn

#Load the Diabetes Dataset
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()
print(diabetes_dataset["DESCR"])

print(diabetes_dataset.keys())

#Save the input and target variables using keys
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

print(targets)

#Normalize the target data (This will make clearer training curves)
targets = (targets - targets.mean(axis=0))/targets.std()
print(targets)

#split the data into train and test splits
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)

#Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1)

    ])

    return model

model = get_model()

model.summary()

#Compile the model
model.compile(optimizer='adam',
              loss = 'mae',
              metrics=['mae'])

#Train the model with some of the data reserved for validation
history = model.fit(train_data, train_targets, epochs=100,
                    validation_split=0.15, batch_size=64, verbose=False)

#Evaluate the model on test set
model.evaluate(test_data, test_targets, verbose=2)

#Plot the training and validation loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epochs')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Training', 'Validation'], loc = 'upper right')
plt.show()

#Model Regularization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

def get_regularised_model(wd,rate):
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu',
              input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(1)
    ])

    return model

#Rebuild the model with weight decay and dropout layers
model = get_regularised_model(1e-05, 0.3)

#Compile the model
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

#Train the model with some of the data reserved for validation
history=model.fit(train_data, train_targets, epochs=100,
                  validation_split=0.15, batch_size=64, verbose=False)

#Evaluate the model on the test set
model.evaluate(test_data,test_targets, verbose=2)

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

#Introduction to callbacks
from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting training....")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: Starting batch {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finishing epoch {epoch}")

    def on_train_end(self, logs=None):
        print(("Finished Training:"))

model = get_regularised_model(1e-5, 0.3)

model.compile(optimizer='Adam',
              loss='mae',
              metrics=['mae'])

model.fit(train_data, train_targets, epochs=3,
          batch_size=128, verbose=False, callbacks=[TrainingCallback()])

class TestingCallback(Callback):
    def on_test_begin(self, logs=None):
        print("Starting Testing...")

    def on_test_batch_begin(self, batch, logs=None):
        print(f"Testing: Starting batch {batch}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"Testing: Finished Batch {batch}")

    def on_test_end(self, logs=None):
        print("Finished Testing:")

print(model.evaluate(test_data,test_targets,verbose=False,
               callbacks=[TestingCallback()]))

class PredictionCallback(Callback):
    def on_predict_begin(self, logs=None):
        print("Starting Prediction....")

    def on_predict_batch_begin(self, batch, logs=None):
        print(f"Prediction: Starting batch {batch}")

    def on_predict_batch_end(self, batch, logs=None):
        print(f"Prediction: Finished batch {batch}")

    def on_predict_end(self, logs=None):
        print("Finished Prediction:")

print(model.predict(test_data,verbose=False,
                    callbacks=[PredictionCallback()]))

#Early Stopping / Patience
#Re-train the unregularised model

unregularised_model = get_model()

unregularised_model.compile(optimizer='adam',
                            loss='mae')

unreg_history = unregularised_model.fit(
    train_data,
    train_targets,
    epochs=100,
    validation_split=0.15,
    batch_size=64,
    verbose=False,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
)

print(unregularised_model.evaluate(test_data, test_targets, verbose=2))

# Re-train the regularised model

regularised_model = get_regularised_model(1e-8,0.2)
regularised_model.compile(optimizer='adam', loss='mae')
reg_history = regularised_model.fit(train_data, train_targets, epochs=100,
                                       validation_split=0.15, batch_size=64,verbose=False,
                                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
# Evaluate the model on the test set

print(regularised_model.evaluate(test_data, test_targets, verbose=2))

#Plot the training and validation loss
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised Model: loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'],
loc = 'upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised Model: Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()



