# **Behavioral Cloning**
Let an DNN take full control of your car ny creating a End-To-End Solution from Sensor input to steering output.

[//]: # (Image References)

[image0]: ./examples/model.png "Original End-To-End Nvidia Model"
[image1]: ./examples/straight.gif "Straight Driving"
[image2]: ./examples/recovery.gif "Recovery Driving"
[image3]: ./examples/flip.jpg "Flip"
[image4]: ./examples/original.jpg "Original"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. At the top of the script you will find options to let the prediction run on CPU or GPU. You can change the steering_multi parameter to increase gain in the control loop. A factor of 4 showed good performance on the first track.

##### Turn autonomous Cycles in the Simulator
```
~/linux_sim/linux_sim.x86_64
python3 drive.py model.h5

```
##### Train the Model
```
python3 clone.py
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 128 (model.py lines 118-124)
The model includes RELU layers to introduce nonlinearity (code line 118 to 135), and the data is normalized in the model using a Keras lambda layer (code line 118).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 130 to 135). The dropouts are only in the fully connected part of the network since this show good performance in the previous projects. The number of epochs was chosen with 1. Since the validation accuracy did not decrease with the second epoch I stopped training to prevent overfitting after the first.

The model was trained and validated on different data sets to ensure that the model was not overfitting by repeating driving behavior. The different styles were **recovery** showing only driving near to the edges and away from them, **sinus** by driving only in sinus agile turns and **straight** by keeping only in the middle of the road. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for multiple turns.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140) with a batch_size of 128 images.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road and doing the hard turns.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to decode the image by multiple Conv2D layers and afterwards flatten this layers for processing them in in combination with dropouts.
I stopped at 5 Conv2D and 4 fully connected layers since the number of parameters was at about 1000000. With a another fully connected layer the network took to long for training. My first step was to use a convolution neural network model similar to the End-To-End approach: [PDF](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

I thought this model might be appropriate because it showed good results in real driving application. After removing the 1164 fully connected layer be GPU was able to train this model fast enough. The following image shows the original architecture:

![alt text][image0]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set 80% to 20%.

To combat the overfitting, I modified the model so that dropouts are integrated at the fully connected end of network. Parts of the data is also augmented in a random way after each epoch.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the hard turns at the end of the track.To improve the driving behavior in these cases,  I used a gain factor on the predicted steering angle of four. The recovery data from both directions also helped to increase the performance in turn situations. The flipping, shifting and dimming of the image also helped to handle this situations. The model showed better performance on turned road than on straight roads.   

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 118-135) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture:

```
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lambda_1 (Lambda)            (None, 90, 320, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8448)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               844900    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                510       
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 981,819
    Trainable params: 981,819
    Non-trainable params: 0
```


#### 3. Creation of the Training Set

To capture good driving behavior, I first recorded two laps on track one using center lane driving in both directions. Here is an example image of center lane driving also called "straight":

![alt text][image1] ![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center line so that the vehicle would learn to bring it back to the center of the road. The images show what a recovery looks like starting from right to the left side. After that I repeated this process into both driving directions and added different driving styles like an agile sinus drive. After the collection process, I had 12786 number of data points containing three images per point. I finally randomly shuffled the data set and put 20% of the data into a validation set. Left and right images have been ignored, since setting a fixed correction factor did not improved the driving behavior.

#### 4. Data Processing
Every image is cropped at the top and bottom, focusing the data only on the street view of the camera.
Inside the generator the images were also randomly dimmed and shifted. To augment the data set, I flipped images and angles thinking that this would equalize the left and right turns. For example, here is an image that has then been flipped:

![alt text][image3] ![alt text][image4]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the validation loss. More epochs showed only very little decrease in the validation set and could show overfitting. I used an adam optimizer so that manually setting the learning rate wasn't necessary.

#### 5. Summery and Improvements
Training the network happened on an embedded Nvidia GPU over night. This took long intervals of about 4 hours but was enough since more epochs did not show any improvements. What really helped was more data from both directions, homogenization with different driving styles and random augmentation of the data. The steering_multi factor helped to lower the stability and increase the agile driving around turns.

A good improvement of the training would be an automated driving with feedback into the training. For that the simulator should deliver the data when the vehicle left the road. It could be done by another DNN that just detects when the vehicle left the road. This would enable a complete unsupervised learning in the simulator.
