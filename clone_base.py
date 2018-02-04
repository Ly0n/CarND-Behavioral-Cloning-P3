import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steerings = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    steering = float(line[3])
    steerings.append(steering)

# Keras needs np array datatype
X_train = np.array(images)
y_train = np.array(steerings)
print(X_train.shape)
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=7,verbose=1)

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)
#
#### print the keys contained in the history object
#print(history_object.history.keys())
#
#### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
#model.save('model.h5')
