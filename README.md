#Behavioral Cloning using Deep Learning
**P3 Submission for Self-Driving Car Nanodegree**
###Overview
The overall methodology followed in this project is as follows:
 1. Use the simulator to collect data of good driving behavior
 2. Build, a convolution neural network in Keras that predicts steering angles from images
 3. Train and validate the model with a training and validation set
 4. Test that the model successfully drives around track one without leaving the road
 5. If there are any "accidents", fine-tune the hyperparameters, the architecture and get more data on corner cases
 
###Deliverables
The following are the files that are included in this project:
 
     * model.py - The script used to create and train the model.
     * drive.py - The script to drive the car. This is a modified version to suit my model.
     * model.json - The model architecture.
     * model.h5 - The model weights.
     * README.md - Project report
     * P3_recording - A recording of the "model" driving the car around the track
     * video.py - Script to convert a series of images to ".mp4" format
     
###Command-Line-Interface
####Training the network
The model can be trained using the following command. The comments in the model.py file explain the code. The model output files are dumped in outputs/steering_model/ to prevent overwriting any existing models in the current directory.

`$ python model.py --data_dir=driving_data --batch=64 --epoch=10`

####Client to send driving signals to the Simulator in Autonomous mode

`$ python drive.py model.json`

###Model Architecture and Training Strategy
I started off with comma.ai Steering Angle Prediction Model. But after a lot of experimentation and tuning, I decided to reduce the number of parameters while making the network deeper. Before I discuss my architecture, I discuss data acquisition and data preprocessing technique that I used. I found that image preprocessing and augmentation played an important role in making sure that the car drives all the way around the track. 

#### 1. Data Acquisiton, Preprocessing and Augmentation
I did not have an analog controller to record training data, and realized that driving-data from keyboard was not smooth. It did not perform well in training the model by itself. So I used the training data released by Udacity in conjunction with the data I collected myself. 
 *I collected mostly recovery data for "wandering off" scenarios. I turned off recording when I let my car wander to the side, and then turned recording ON while I gracefully steered to the center. I repeated this for all the sharp turns as well.
 #Since the track is mostly left-turn biased, I also collected recovery-data in the opposite direction to balance it out.

The simulator dumps references to the image files as well as the telemetry data into driving_log.csv file. For each frame-steering angle, there is an image each from center, left and right cameras. The left and right cameras were used as extra training data for scenarios where the car is off-center. Since the steering angle provided is actual w.r.t center, bias needs to be added for left/right to direct the vehicle to ground-truth.

   **For left camera image**: I added 0.15 offset (add bias to steer right) to the steering angle
   
   **For right camera image**: I subtracted 0.15 offset (add bias to steer left) to the steering angle
   
Also, since the track is l
