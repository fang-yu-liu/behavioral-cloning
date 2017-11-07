## Project: Behavioral Cloning

Overview
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./write-up-data/image_flipping.png "Image flipping"

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: The script to create and train the model.
* model.ipynb: The jupyter notebook version of model.py (Reference only). model.py was converted from this file.
* drive.py: The script for driving the car in autonomous mode.
* model.h5: The trained convolution neural network.
* read_data.ipynb: The initial jupyter code for reading image data and augmentation. Changed to use generator in model.ipynb file later on. (Reference only)
* video.py: The script for creating video from images.
* video.mp4: The recorded video of the vehicle driving autonomously using the trained model for one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```python drive.py model.h5```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. It can be run by ```python model.py```. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model was based on the NVIDIA architecture (https://arxiv.org/abs/1604.07316). It consists the following 12 layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         	| 160x320x3 RGB image   |
| Normalization   |  |
| Cropping        |  |
| Convolution 5x5 | 24 filters, 2x2 stride, relu activation 	|					
| Convolution 5x5	| 36 filters, 2x2 stride, relu activation     									|
| Convolution 5x5 | 48 filters, 2x2 stride, relu activation |
| Convolution 3x3 | 64 filters, 1x1 stride, relu activation |
| Convolution 3x3 | 64 filters, 1x1 stride, relu activation |
| Flatten	|       									|
| Fully connected		|	outputs 100, relu activation   |
| Dropout	      	| keep_prob 0.5 				|
| Fully connected		| outputs 50, relu activation   |
| Dropout	      	| keep_prob 0.5 				|
| Fully connected		| outputs 10, relu activation |
| Dropout	      	| keep_prob 0.5 				|
| Fully connected		| outputs 1, relu activation     |

1. Normalization layer: ```model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
``` Normalize the input image
2. Cropping layer: ```model.add(Cropping2D(cropping=((70,25),(0,0))))
``` The upper part of the image contains the sky and trees and the lower part of the image contains the hood of the car. Both of those are irrelevant informations for the model. Therefore, cropping out those parts will help the model focus on learning the useful part in the image.
3. Implemented 5 convolution layers based on the NVIDIA architecture.
4. Implemented 4 fully connected layers based on the NVIDIA architecture.
5. Added dropout layers after each fully connected layers to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The following steps are implemented to avoid overfitting:
1. Data augmentation: Images were augmented by flipping the images to make it twice the size of the initial data set.
2. Dropout layers: 3 Dropout layers with keep_prob = 0.5 were added after each fully connected layers. Since originally the mean squared error was higher on the validation set while the mean squared error was low on the training set.

#### 3. Model parameter tuning

An adam optimizer was used so that manually tuning the learning rate wasn't necessary (model.py line 122).

#### 4. Appropriate training data

The sample data provided by Udacity was used for training the model. First, tried to use only the center images, the car seemed to drive towards the edge while turning and was unable to recover back to the center of the track. Adding right and left images with steering correction by 0.2 to help the model learn how to recover from the right and left side of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First, a two-layer neural network was implemented to verify the whole setup works and the car can drive autonomously using the trained model. Only center images/measurements in the training data were used. The car was able to drive autonomously but was stuck on the bridge because it drove out of bound while turning.
Second attempt was changing the model to use the LeNet architecture. The result of the second model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To avoid overfitting, I augmented the data by flipping the images to increase the data set size. Also, to help the model learn how to recover from the side of the track, I included the left and right images in training data set and added 0.2 steering corrections for the left and right steering measurements. The car was able to drive further in the track 1 but still fell off the track on a few spots. To improve the driving behavior, I changed the model to use the NVIDIA architecture, and added dropout layers to further avoid overfitting. At the end of the process, the car is able to drive autonomously around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 105-120) was based on the NVIDIA architecture (https://arxiv.org/abs/1604.07316) with small modification. (See the first section of Model Architecture and Training Strategy for details of the model)

#### 3. Creation of the Training Set & Training Process

* Training data set: Udacity sample training data.
* Validation data set: Split from the training data set. (0.2 of the training data). Validation set was used to help determine if the model was over or under fitting.
* Epoch numbers: 5
* Steering correction: Center images and their steering measurements were used directly. Left and right images were used to tell the model how to recover from the left and right side of the track. Therefore, a steering correction (0.2) was added to the left and right measurements.
* Data augmentation: Augmenting the training and validation data set by flipping the images and steering measurements to double the data set size.
The following shows the image before and after flipping:
![Image augmentation][image1]

* Data shuffling: The images and measurements data were shuffled before feeding to the neural network.
