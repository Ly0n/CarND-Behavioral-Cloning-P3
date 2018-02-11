import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import sklearn
import os
import cv2
# Run on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set your Hyperparameters
EPOCHS = 1
batch_size = 64
learning_rate = 0.0001
cropx_top = 50
cropx_bottom = 140
x_new = cropx_bottom - cropx_top
y_new = 320

# Get your Samples
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Define Preprocessing Pipeline
def crop_image(image):
    cropped = image[cropx_top:cropx_bottom,0:y_new]
    return cropped

def change_brightness(image):
    image = image + np.random.random_integers(-20,20)
    return image

def normal_image(image):
    image = (image / 255.0) - 0.5
    return image

def process_image(image):
    image = change_brightness(image)
    image = normal_image(image)
    image = crop_image(image)
    return image

# Include Datashuffle and train split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples,batch_size=128):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
                 batch_samples = samples[offset:offset+batch_size]
                 images = []
                 steerings = []
                 for batch_sample in batch_samples:
                    filename1 = batch_sample[0].split('/')[-1]
                    filename2 = batch_sample[1].split('/')[-1]
                    filename3 = batch_sample[2].split('/')[-1]
                    current_path = './data/IMG/'
                    correction = 0.25
                    steering_center = float(batch_sample[3])
                    # Dont use the Data with 0 Steering
                    if abs(steering_center) < 0.01:
                        #print("Bin the Data")
                        continue

                    rand_image = np.random.randint(3)

                    if rand_image == 0:
                        image = process_image(cv2.imread(current_path + filename1,cv2.IMREAD_COLOR))
                        if np.random.randint(2) == 1:
                                steerings.append(steering_center)
                                image = np.fliplr(image)
                                images.append(image)
                        else:
                            images.append(image)
                            steerings.append(steering_center)

                    if rand_image == 1:
                        # Left camera
                        image = process_image(cv2.imread(current_path + filename2,cv2.IMREAD_COLOR))
                        if np.random.randint(2) == 1:
                            steerings.append(-(steering_center - correction))
                            image = np.fliplr(image)
                            images.append(image)
                        else:
                            images.append(image)
                            steerings.append(steering_center - correction)

                    if rand_image == 2:
                        # Left Camera
                        image = process_image(cv2.imread(current_path + filename3,cv2.IMREAD_COLOR))
                        if np.random.randint(2) == 1:
                        #
                            steerings.append(-(steering_center + correction))
                            image = np.fliplr(image)
                            images.append(image)
                        else:
                            images.append(image)
                            steerings.append(steering_center + correction)

                    # Keras needs np array datatype
                    X_train = np.array(images)
                    y_train = np.array(steerings)

                    yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Conv2D,Dropout
from keras.regularizers import l2

model = Sequential([

    # 5 Conv2D Layers with weight regularizer
    # It draws samples from a truncated normal distribution centered on 0 with
    # Activation Function elu
    # Low Droput

    Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal',input_shape=(x_new, y_new, 3)),
    Dropout(.1),
    Conv2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal'),
    Dropout(.2),
    Conv2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal'),
    Dropout(.2),
    Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal'),
    Dropout(.2),
    Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal'),
    Flatten(),

    # 5 Fully Connected Layers
    # Dropout with drop probability of .5

    # Added another dense layer
    Dense(500, activation='elu', init='he_normal'),
    Dropout(.5),

    Dense(100, activation='elu', init='he_normal'),
    Dropout(.5),
    Dense(50, activation='elu', init='he_normal'),
    Dropout(.5),
    Dense(10, activation='elu', init='he_normal'),
    Dropout(.5),        # Output
    Dense(1, activation='linear', init='he_normal')
    # One single steering output
])

model.summary()
model.compile(loss='mse', optimizer='Adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS,verbose=1)

model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')# Not Prints since docker :D
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_history.png')

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
