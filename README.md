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
I started off with comma.ai Steering Angle Prediction Model. But after a lot of experimentation and tuning, I decided to reduce the number of parameters while making the network deeper. Initially the validation loss was much worse than the training loss. So I added more dropout layers in the Fully-Connected-Layers. This helped reduce **overfitting**. Before I discuss my architecture, I discuss data acquisition and data preprocessing technique that I used. I found that image preprocessing and augmentation played an important role in making sure that the car drives all the way around the track. 

#### 1. Data Acquisiton, Preprocessing and Augmentation
I did not have an analog controller to record training data, and realized that driving-data from keyboard was not smooth. It did not perform well in training the model by itself. So I used the training data released by Udacity in conjunction with the data I collected myself. 

--> I collected mostly recovery data for "wandering off" scenarios. I turned off recording when I let my car wander to the side, and then turned recording ON while I gracefully steered to the center. I repeated this for all the sharp turns as well.
    
--> Since the track is mostly left-turn biased, I also collected recovery-data by driving in the opposite direction to balance it out.

The simulator dumps references to the image files as well as the telemetry data into driving_log.csv file. For each frame-steering angle, there is an image each from center, left and right cameras. The left and right cameras were used as extra training data for scenarios where the car is off-center. Since the steering angle provided is actual w.r.t center, bias needs to be added for left/right to direct the vehicle to ground-truth.

   **For left camera image**: I added 0.15 offset (add bias to steer right) to the steering angle
   
   **For right camera image**: I subtracted 0.15 offset (add bias to steer left) to the steering angle
   
--> From the training data I collected, I discarded the "center" images with steering angle = 0, since it does not give much information. I already rely on udacity data for straight-driving behavior. I keep the left and right image counterparts with 0.15 offset.
   
--> To generate more data and to remove bias towards any particular direction, I flipped 50% of the images with abs(steering angle) > 0.1, and added them to the training data.

--> I cropped each image from the top to remove un-interesting artifacts such as trees etc. The entire width of the image was preserved.

--> I used cv2.resize w/ INTER_AREA interpolation to resize all images to 32x32x3 (RGB format).

--> I also experimented with Sobel filtering, but it did not give any improvement. So I dropped the idea.

#### 2. Model Architecture

The following is the architecture I finally arrived at. It was coded using Keras (w/ TensorFlow). The validation loss

**Input:** The model accepts 32x32x3 (RGB format) data

**Layer 0:** 1x1 2D Convolution w/ ELU activation (This normalizes the colorspace)

**Layer 1:** 3x3x32 2D Convolution w/ 2x2 Maxpool and ELU activation

**Layer 2:** 3x3x64 2D Convolution w/ 2x2 Maxpool and ELU activation

**Layer 3:** 3x3x128 2D Convolution w/ ELU activation and **dropout**

**Flatten:** 1024 output

**Layer 4:** Fully Connected Layer -- 1024 output w/ **dropout (0.5)** and ELU activation

**Layer 5:** Fully Connected Layer -- 512 output w/ **dropout (0.5)** and ELU activation

**Layer 4:** Fully Connected Layer -- 128 output w/ **dropout (0.5)** and ELU activation

**Output:** Fully Connected with 1 output value


#### 3. Training

After preprocessing and augmentation, I had ~35k images to train with. The Adam optimizer was used with mean-squared error loss. Concatenated data from udacity and my simulator was passed onto the model with 20% of the data as validation data. After some experiments, I settled with 10 epochs and 64 batch_size. The data was shuffled for every epoch. If I ran more epochs, it led to car driving off the cliff.

#### 4. Simulation

The drive.py script was used to evaluate the model trained above. It interacts with the simulator by grabbing center camera images, passing through the model, and returning the telemetry data (steering angle and throttle). The throttle is hard-coded as 0.2. I modified drive.py to crop and resize the streaming images in the same way as while training.

**RESULT: The car drove correctly on track 1 without driving off the road**

I am attaching the recording of my model driving the car.


