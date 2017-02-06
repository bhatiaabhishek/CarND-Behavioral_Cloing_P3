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

###Model Architecture



