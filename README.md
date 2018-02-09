# **Deep Imitation Learning for a Self-Driving Car**

## Project Goal

---

The goal of this project was to teach a car to navigate on various road simulations in different conditions (such as flat or hilly terrain and with bright or dark lighting). If the model was able to successfully drive the car through a complete lap around the road, then the model was successful.

### Final Model Results
**Racetrack Navigation**: https://www.youtube.com/watch?v=hfr8NDPdqMI

**Mountainous Terrain Navigation**: https://www.youtube.com/watch?v=M-V56iYyFkE


[//]: # (Image References)


[training_images]: ./examples/training_data.png
[transformed_images]: ./examples/transformed_data.png
[distributional_drift]: ./examples/distributional_drift.png
[architecture]: ./model_architecture.png

---

### Training Strategy

The overall strategy was inspired by the NVIDIA research paper, [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/pdf/1704.07911.pdf).

The training data is comprised of 61,488 frames from the three front-facing cameras on the simulated car (each camera produced 20,496 frames). The validation set consists of 15,375 frames (5,125 frames per camera). Thus, in aggregate, the data set consists of 76,863 frames that were then randomly shuffled with an 80/20 split. The data was collected by driving eight laps around each road simulation with four laps going clockwise and four laps going counter-clockwise. This technique was used to help prevent overfitting by ensuring the car is able to experience turns and curves in different directions. Another alternative would be to flip the frame, but this was not used since I wanted the car to have some adherence to lane lines, and a flipped frame's orientation would have interfered with that behavior. 

Below is a sample of the training data. The floating point X-label is the original steering angle (the model's target variable) that was present at the time the frame was recorded.

![alt text][training_images]

Notice the obvious differences in terrain and lighting between the frames. The deep neural network could have picked up on these differences and memorized a correlation between those characteristics and a particular steering angle, thus failing to adapt to different road conditions and overfitting. One way to prevent this from happening with the terrain characteristics would have been to crop the image to focus exclusively on the road. However, it was not clear that the NVIDIA research team utilized this technique and my model's driving ability was not impacted by not cropping the frames (although doing so would have reduced the model's training time). I did apply a brightness transformation to each image in the training data using a Gaussian normal distribution with a 0.5 mean and standard deviation of 0.35. This generated an additional five new frame for each training image, thereby resulting in a total of 368,928 frames in the training set.

Here is a sample of the transformed data:

![alt text][transformed_images]

#### Handling Distributional Drift

End-to-end imitation learning is based on the premise that rather than relying upon feature engineering with computer vision and sensor fusion, why not make the problem of autonomy into a supervised learning problem?  The training data is genereted by an expert (in the case of a self-driving car, the driver), and the optimization goal is for the model to act just like the expert.  This sounds like a great idea! Unfortunately, what happens if the model makes a small mistake?  That mistake means that the model is no longer behaving like the expert.  Even in the case of a small mistake, its impact could grow and grow as the model continues to act in the context of having made such a mistake. For a self-driving car, imagine if the car drifted a bit to the side of the road and continued to move in that same direction. Eventually, it would run off the road. Since the training data has its own distribution, and the model is acting upon test data, which also has its own distribution, this growing difference between the two distributions is called distributional drift.  Is there anyway to solve it for a self-driving car?

![alt text][distributional_drift]

One option would be to collect data from the car by having the expert repeatedly cause the car to drift on the road, and then recover.  Obviously, in real-life this would be rather dangerous, and even in a simulator it can be difficult to replicate.  Plus, you need a lot of imagery to demonstrate these techniques.

Fortunately, this is exactly why the training data does not only contain frames recorded directly from the center-forward orientation of the car, but also from the left and right. As you can see from the above imagery examples, the car's perspective changes depending on which camera image you are viewing. Notice how the left camera's images are similar to what you would expect to see if the car had drifted to the left.  Likewise, the right camera's images are similar to what you would expect to see if the car had drifted to the right. Eureka! If we just add small positive values to the left camera's corresponding steering angles, the car would learn to turn to the right if it has drifted to the left. And, if we just add small negative values to the right camera's angles, the car would learn to turn to the left if it has drifted to the right. These simple alterations are then enough to handle the distributional drift problem (at least in this case). We now have everything we need for the training set.

---

## Model Architecture & Training

The end-to-end deep convolutional neural network consists of eleven layers with 8,404,119 trainable parameters. The output layer does not have an activation function and only has one unit since this network is being used for a regression task (in this case, to predict the steering wheel angle).

| Layer         		|     Description	        					| Activation | Output Dimensions |
| :-------------------- |:---------------------------------------------:|:----------:|:-----------------:| 
| Input         		| 160x320x3 RGB image   						|            | 160x320x3         |
| 5x5 Convolution     	| 2x2 stride w/ valid padding                	| RELU       | 78x158x24         |
| 5x5 Convolution     	| 2x2 stride w/ valid padding                	| RELU       | 37x77x36          |
| 5x5 Convolution     	| 2x2 stride w/ valid padding                	| RELU       | 17x37x48          |
| 5x5 Convolution     	| 2x2 stride w/ valid padding                	| RELU       | 15x35x64          |
| 5x5 Convolution     	| 2x2 stride w/ valid padding                	| RELU       | 13x33x64          |
| 3x3 Convolution	    | 1x1 stride w/ same padding      				| RELU       | 8x8x256           |
| Flatten        	    |                                               |            | 1x27456           |
| Fully-connected  	    | 300 units                                     | RELU       | 1x300             |
| Fully-connected  	    | 100 units                                     | RELU       | 1x100             |
| Fully-connected  	    | 50 units                                      | RELU       | 1x50              |
| Fully-connected  	    | 1 units                                       |            | 1x50              |

Or for another look:

![alt text][architecture]

I added an additional fully-connected layer (the one after the convolutions have been flattened) in an effort to increase the representational capactiy a bit more than the model. It should also be noted that the input image being processed is larger than NVIDIA's input size of 66x200 px. My decision was based purely off the desire to maintain the higher resolution for recording the model's behavior. Otherwise, I would have preferred the smaller input size to reduce training time.

The model was trained on the 368,928 frames in the training set (as described earlier) using an Adam optimizer with a learning rate of 0.001. Early stopping was also utilized to help prevent overfitting from training the model for too long. A mean-squared error (MSE) loss function was used (since this is a regression problem). After training the model for three epochs, it had an MSE loss of 0.0405. The model could have been trained longer since it is underfitting (the training loss is 0.0471), but the current model is sufficient to complete the simulated roads. As a result, I did not use any additional methods to prevent overfitting.