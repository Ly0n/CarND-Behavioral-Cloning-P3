import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import sklearn
import os
import cv2

# More tools for debugging an image processing
from utils import crop_image,normal_image,process_image
from debug import ipsh

# Run on CPU or comment out for GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# GLobal Image Keeper

image_orgi = 0
imageshift = 0
# Set your Hyperparameters
EPOCHS = 1
batch_size = 128 # Higher Batchsize is not possible with 2GB RAM
#learning_rate = 0.0001
cropx_top = 50
cropx_bottom = 140
x_new = cropx_bottom - cropx_top
y_new = 320
arg_div = 18 # Increasing the argumentation div will lower the probability for image argumentation

# Get your Samples
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
# Datashuffle and train split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
ipsh() # Embedded ipython console
def generator(samples,batch_size=128):
    num_samples = len(samples)
    while 1:
        # Shuffle the trainings data
        shuffle(samples)
        # Split the data in batches
        for offset in range(0,num_samples,batch_size):
                 batch_samples = samples[offset:offset+batch_size]
                 images = []
                 steerings = []
                 # Process every batch
                 for batch_sample in batch_samples:
                    # Get the front camera image
                    filename1 = batch_sample[0].split('/')[-1]
                    """
                    # Load images from car side cameras
                    filename2 = batch_sample[1].split('/')[-1]
                    filename3 = batch_sample[2].split('/')[-1]
                    """
                    # Get images from fixed path that is not the same as csv record
                    current_path = './data/IMG/'
                    steering_center = float(batch_sample[3])
                    """
                    # Not needed when training data is generated with mouse
                    if abs(steering_center) < 0.01:
                        print("Jump over straight data")
                        continue
                    """
                    #Load the Images
                    image = cv2.imread(current_path + filename1)
                    # DNN shows good results in the YUV colorspace
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                    # Just do the cropping for all images since normalization
                    image = crop_image(image)

                    # Flip Image and steering
                    if np.random.randint(arg_div) == 1:
                        steering_center = -steering_center
                        image = np.fliplr(image)

                    # Change the brightness of the images
                    if np.random.randint(arg_div) == 1:
                        image = image + np.random.random_integers(-20,20)

                    # Shift the images and the steerings
                    if np.random.randint(arg_div) == 1:
                        dx=15
                        dy=15
                        # Randomly change the shifting
                        sx = dx * (np.random.rand() - 0.5)
                        sy = dy * (np.random.rand() - 0.5)
                        # Steering angle shifter
                        steering_center += sx * 0.002
                        mask = np.float32([[1, 0, sx], [0, 1, sy]])
                        height, width = image.shape[:2]
                        image = cv2.warpAffine(image, mask, (width, height))

                    # Append training data
                    steerings.append(steering_center)
                    images.append(image)

                    # Keras needs np array datatype
                    X_train = np.array(images)
                    y_train = np.array(steerings)

                    # Shuffle the training data before return to the generator
                    yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Conv2D,Dropout

model = Sequential()

    # 5 Conv2D Layers with weight regularizer
    # It draws samples from a truncated normal distribution centered on 0 with
    # Activation Function elu
    # Low Droprelu
model.add(Lambda(lambda x: x / 255.0 -0.5,input_shape=(x_new, y_new, 3)))
model.add(Conv2D(24, 5, 5, activation='relu',subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu',subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu',subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu',subsample=(1, 1)))
model.add(Conv2D(64, 3, 3, activation='relu',subsample=(1, 1)))
model.add(Flatten())

    # 5 Fully Connected Layers
    # Dropout with drop probability of .5 and .25

model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.25))        # Output
model.add(Dense(1, activation='linear'))
    # One single steering output as a linear function


model.summary()
model.compile(loss='mse', optimizer='Adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS,verbose=1)

model.save('model.h5')

ipsh()
# Saving the figure does not work form the docker container
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
fig.title('model mean squared error loss')
fig.ylabel('mean squared error loss')
fig.xlabel('epoch')
fig.legend(['training set', 'validation set'], loc='upper right')
fig.savefig('loss_history.png')
fig.close(fig)

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validatio
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
