import csv
import cv2
import numpy as np

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

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(3,160,320))
model.add(lambda x: x / 255.0 - 0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=7,verbose=1)

model.save('model.h5')
