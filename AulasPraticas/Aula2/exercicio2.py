import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print(50*'*'+' Parte 2 '+50*'*')
#tensorflow version being used
print('TF version:', tf.__version__)

#is tf executing eagerly? 
print(tf.executing_eagerly())

#load mnist training and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#we have 10 labels (0:Zero, 1:One, 2:Two, 3:Three, 4:Four, 5:Five, 6:Six, 7:Seven, 8:Eight, 9:Nine)
#each image is mapped to one single label. class names are not included in the dataset
labels = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

#data shape and cardinality
print('Train set shape', x_train.shape)
print('Train labels shape', y_train.shape)
print('Test set shape', x_test.shape)
print('Test labels shape', y_test.shape)
print('Number of training samples', len(x_train))
print('Number of testing samples', len(x_test))

#show a figure
# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#plotting some numbers!
# plt.figure(figsize=(8,8))
# for i in range(25):
#     plt.subplot(5,5,i+1) #Add a subplot as 5 x 5 
#     plt.xticks([]) #get rid of labels
#     plt.yticks([]) #get rid of labels
#     plt.grid(False)
#     plt.imshow(x_test[i], cmap="gray")
#     plt.xlabel(labels[y_test[i]])
    
# plt.show()

#reshape the input to have a list of 784 (28*28) and normalize it (/255)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) 
x_train = x_train.astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]) 
x_test = x_test.astype('float32')/255

#building a three-layer sequential model
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')
])

#compiling the model
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#training it
model.fit(x_train, y_train, epochs=10, batch_size=256)

#evaluating it
print(50*'*'+' Accuracy '+50*'*')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2, batch_size=256)
print('\nTest accuracy:', test_acc)

#finally, generating predictions (the output of the last layer)
print('\nGenerating predictions for the first fifteen samples...') 
predictions = model.predict(x_test, batch_size=256)

#prediction 0
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_test[0])
print(labels[y_test[0]])

print('Predictions shape:', predictions.shape)
error = 0
total = 0
for i, prediction in enumerate(predictions):
    #tf.argmax returns the INDEX with the largest value across axes of a tensor
    predicted_value = tf.argmax(prediction)
    if predicted_value.numpy() != y_test[i]:
        error = error + 1
    label = labels[y_test[i]]
    print('Predicted a %d. Real value is %s.' %(predicted_value, label))
    total = total + 1 

correct = total-error
accuracy = (correct/total) * 100
print('Total samples: %d, Correct: %d, Incorrect: %d, Accuracy: %f' %(total, correct, error, accuracy))
