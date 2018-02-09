import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import sklearn
from keras.regularizers import l2

EPOCHS = 3
cropx_start = 20
cropx_stop = 110
x_new = cropx_stop - cropx_start
y_new = 320
batch_size = 256

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def crop_image(image):
    cropped = image[cropx_start:cropx_stop,0:y_new]
    return cropped

def flip_image(image):
    if np.random.randint(2) == 1:
        image = np.fliplr(image)
    return image

def process_image(image):
    cropped_image = crop_image(image)
    flipped_image = flip_image(cropped_image)
    return flipped_image

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
                    correction = 0.2
                    steering_center = float(batch_sample[3])
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    rand_image = np.random.randint(3)
                    if rand_image == 0:
                    images.append(process_image(imageio.imread(current_path + filename1)))
                    steerings.append(steering_center)

                    if rand_image == 1:
                    images.append(process_image(imageio.imread(current_path + filename2)))
                    steerings.append(steering_left)

                    if rand_image == 2:
                    images.append(process_image(imageio.imread(current_path + filename3)))
                    steerings.append(steering_right)

                    # Keras needs np array datatype
                    X_train = np.array(images)
                    y_train = np.array(steerings)

                    yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Conv2D,Dropout

model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(x_new, y_new, 3)),
    # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation
    Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .1 (keep probability of .9)
    Dropout(.1),
    # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
    Conv2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .2 (keep probability of .8)
    Dropout(.2),
    # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
    Conv2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .2 (keep probability of .8)
    Dropout(.2),
    # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
    Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .2 (keep probability of .8)
    Dropout(.2),
    # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
    Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal',W_regularizer=l2(0.001)),
    # Flatten
    Flatten(),
    # Dropout with drop probability of .3 (keep probability of .7)
    Dropout(.3),
    # Fully-connected layer 1 | 100 neurons | elu activation
    Dense(100, activation='elu', init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .5
    Dropout(.5),
    # Fully-connected layer 2 | 50 neurons | elu activation
    Dense(50, activation='elu', init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .5
    Dropout(.5),
    # Fully-connected layer 3 | 10 neurons | elu activation
    Dense(10, activation='elu', init='he_normal',W_regularizer=l2(0.001)),
    # Dropout with drop probability of .5
    Dropout(.5),        # Output
    Dense(1, activation='linear', init='he_normal',W_regularizer=l2(0.001))
])

model.summary()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
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
