#**Neilkunal Panchal** 

#**Behavioral Cloning** 

##Writeup Template

###Goals

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

A number of Architectures were tried. The first architecture was LeNet. Lasty the CNN model from the nVidia group was used.

####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

```
|Layer (type)             |        Output Shape      |    Param #   |  Connected to    |                
|=========================|===========================|=============|==================|
|lambda_1 (Lambda)        |        (None, 160, 320, 3)  | 0          | input[0][0]     |        
|_________________________|___________________________|______________|_________________|
|cropping2D_1 (Cropping2D)|       (None, 90, 320, 3)  |  0           |lambda_1[0][0]  |                 
|_________________________|___________________________|______________|_________________|
|convolution2d_1 (Conv2D) | (None, 43, 158, 24)  | 1824     |   cropping2d_1[0][0]     |          
|_________________________|______________________|___________|_________________________||convolution2d_2 (Conv2D) | (None, 20, 77, 36)   | 21636     |  convolution2d_1[0][0] |           
|_________________________|______________________|___________|_________________________|
|convolution2d_3 (Conv2D)  |(None, 8, 37, 48)   |  43248      | convolution2d_2[0][0] |           
|__________________________|____________________|_____________|________________________|
|convolution2d_4 (Conv2D) | (None, 6, 35, 64)   |  27712 | convolution2d_3[0][0]|           |_________________________|_____________________|_____________|________________________|
|flatten_1 (Flatten)       |    (None, 8448)     |     0       | convolution2d_5[0][0] | 
|__________________________|_____________________|_____________|_______________________|
|dropout_1 (Dropout)        |        (None, 8448) |      0   |   flatten_1[0][0]    | 

|__________________________|_____________________|_____________|_______________________|
|dense_1 (Dense)          |        (None, 100)  |      844900   |   dropout_1[0][0]    | |__________________________|_____________________|_____________|_______________________|
|dropout_2 (Dropout)        |        (None, 100) |      0   |   dense_1[0][0]    | 
            
|_________________________|______________________|______________|______________________|
|dense_2 (Dense)          |        (None, 50)    |        5050  |      dropout_2[0][0]   |                 
|_________________________|______________________|______________|______________________|
|dense_3 (Dense)       |           (None, 10)    |        510   |      dense_2[0][0]   |                 
|_______________________|________________________|______________|______________________|
|dense_4 (Dense)      |            (None, 1)     |        11     |     dense_3[0][0]   |                 
|=====================|==========================|===============|=====================|
    
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Furthermore the number of epochs of training was chosen as a method of regularisation. It was found that by using the nVidia arcitecture without dropout, 3 epochs was sufficient. with the 2 drop out laters, 5 epochs was sufficient. After which increasing for any longer resulted in an increase in the validation error.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Furthermore driving backwards on the road to remove the left turn bias, in addition to using the three cameras on the car. Certain parts of the road were prone during testing to veer the car off trajectory. During training, additional data was collected specifically in these problem turns.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to try the LeNet model. This involved training with three epochs. This allowed the testing of the architecture however the results were not satisfactory for driving around the track, as the car would struggle to stay on track.

The following steps were introduced which improved training:

1. Introducing a Lambda layer to normalise and centre the data.
2. crop the images to remove the hood of the car and scenery in the background leaving only the road for training

The next part involved the use of a better model. The model used was from the nVidia group. 

The following modifications were made to improve performance on this model

1. This model is described above and modified for a final fully connected layer to have one output in preparation for steering wheel angle data.
2. Augmented the data by adding the same image flipped with a negative angle. 



####2. Final Model Architecture

The final model architecture (model.py ) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

```
|Layer (type)             |        Output Shape      |    Param #   |  Connected to    |                
|=========================|===========================|=============|==================|
|lambda_1 (Lambda)        |        (None, 160, 320, 3)  | 0          | input[0][0]     |        
|_________________________|___________________________|______________|_________________|
|cropping2D_1 (Cropping2D)|       (None, 90, 320, 3)  |  0           |lambda_1[0][0]  |                 
|_________________________|___________________________|______________|_________________|
|convolution2d_1 (Conv2D) | (None, 43, 158, 24)  | 1824     |   cropping2d_1[0][0]     |          
|_________________________|______________________|___________|_________________________||convolution2d_2 (Conv2D) | (None, 20, 77, 36)   | 21636     |  convolution2d_1[0][0] |           
|_________________________|______________________|___________|_________________________|
|convolution2d_3 (Conv2D)  |(None, 8, 37, 48)   |  43248      | convolution2d_2[0][0] |           
|__________________________|____________________|_____________|________________________|
|convolution2d_4 (Conv2D) | (None, 6, 35, 64)   |  27712 | convolution2d_3[0][0]|           |_________________________|_____________________|_____________|________________________|
|flatten_1 (Flatten)       |    (None, 8448)     |     0       | convolution2d_5[0][0] | 
|__________________________|_____________________|_____________|_______________________|
|dropout_1 (Dropout)        |        (None, 8448) |      0   |   flatten_1[0][0]    | 

|__________________________|_____________________|_____________|_______________________|
|dense_1 (Dense)          |        (None, 100)  |      844900   |   dropout_1[0][0]    | |__________________________|_____________________|_____________|_______________________|
|dropout_2 (Dropout)        |        (None, 100) |      0   |   dense_1[0][0]    | 
            
|_________________________|______________________|______________|______________________|
|dense_2 (Dense)          |        (None, 50)    |        5050  |      dropout_2[0][0]   |                 
|_________________________|______________________|______________|______________________|
|dense_3 (Dense)       |           (None, 10)    |        510   |      dense_2[0][0]   |                 
|_______________________|________________________|______________|______________________|
|dense_4 (Dense)      |            (None, 1)     |        11     |     dense_3[0][0]   |                 
|=====================|==========================|===============|=====================|
    
```





####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/center_2017_09_14_08_48_13_132.jpg)





I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/left_2017_09_14_09_20_38_377.jpg)
![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/left_2017_09_14_09_20_38_719.jpg)

![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/left_2017_09_14_09_20_39_061.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/flipped0.jpg)
![alt text](/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/flipped.jpg)



Data was collected on the first track whilst driving in both directions, and veering back on track.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the number of epochs after which the validation error started to increase. With the dropout layers the number of epochs increased to 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
