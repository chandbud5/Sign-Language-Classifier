import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Importing Dataset
df = pd.read_csv("D:\Datasets\Sign Language Mnist\sign_mnist_train.csv")
test_df = pd.read_csv("D:\Datasets\Sign Language Mnist\sign_mnist_test.csv")

# Preparing Dataset
# For training
df_numpy = np.asanyarray(df)

train_X = df_numpy[:, 1:]
train_Y = df_numpy[:, 0]

train_X = train_X.reshape(27455, 28, 28, 1)
train_Y = train_Y.reshape(27455, 1)

train_X = train_X/255.0

plt.imshow(train_X[45].reshape(28, 28))
plt.show()


# For testing
df_test_numpy = np.asanyarray(test_df)

test_X = df_test_numpy[:, 1:]
test_Y = df_test_numpy[:, 0]

test_X = test_X.reshape(7172, 28, 28, 1)
test_Y = test_Y.reshape(7172, 1)

test_X = test_X/255.0

plt.imshow(test_X[909].reshape(28, 28))
plt.show()


# Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callme = myCallback()

# model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(75, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(25, activation=tf.nn.softmax)
])


# Augmentation
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.2,
                                                            width_shift_range=0.2,
                                                            zoom_range=0.2,
                                                            rotation_range=0.2,
                                                            shear_range=0.2)

# Compiling Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# Fitting model
history = model.fit(train_gen.flow(train_X, train_Y, batch_size=125),
                    epochs=20,
                    steps_per_epoch=220,
                    validation_data=(test_X, test_Y),
                    callbacks=[callme],
                    verbose=1)

# Prediction
pred = model.predict(test_X)
label = np.argmax(pred,axis=1)
print(label.shape,label)
np.savetxt("submission.csv", label)

# Confusion matrix
cm = confusion_matrix(train_Y, pred)
cmap = plt.get_cmap('Blues')
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=cmap)
tick_marks = np.arange(25)
plt.xticks(tick_marks, tick_marks, rotation=45)
plt.yticks(tick_marks, tick_marks)
plt.xlabel("Predicted class")
plt.ylabel("True Label")
plt.savefig("Confusion-Matrix.png", dpi=300)
plt.show()