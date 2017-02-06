import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.activations import relu, softmax, elu
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
from scipy.misc import imread, imsave, imresize
from scipy import ndimage
import cv2
import numpy as np
import time
import random


def load_data(data_dir,data_dict,udacity_data_ld):

    data_pd = pd.read_csv(data_dir + "/driving_log.csv",header=None)

    images = []
    y_data = []
    for idx in range(0,data_pd.shape[0]):
        for camera_ang in ['center','left','right']:
            file_name = data_pd[data_dict[camera_ang]][idx]
            file_path = file_name.strip() if (udacity_data_ld == 0) else ("data/" + file_name.strip())
            steer_ang =  data_pd[data_dict['steering']][idx]
            offset = 0
            if (camera_ang == 'left'):
                offset = 0.15
            elif (camera_ang == 'right'):
                offset = -0.15
            elif ((steer_ang == 0) and (udacity_data_ld == 0)): # If the angle is 0 then exclude it 
                continue
            angle = steer_ang + offset
            im1 = imread(file_path).astype(np.float32)
            im_x,im_y,ch = im1.shape

            # Cropping Image
            im_crop = im1[40:150,0:320,:]

            # Resize Image
            im_resize = cv2.resize(im_crop,(32,32),interpolation=cv2.INTER_AREA)
            
            # Append the processed images
            images.append(im_resize)
            y_data.append(angle)
 
            # Since most of the track is left-turn biased, flip the images 50% of the time if the steering angle is large
            if ((random.random() >= 0.5) and  (abs(steer_ang) > 0.1)):
                imflip = np.fliplr(im_resize)
                images.append(imflip)
                y_data.append(-angle)
         
    X_data = np.array(images)
    y_data = np.array(y_data) 
    
   
    return X_data, y_data    


def my_model3():
    ch = 3
    h = 32
    w = 32 
    model = Sequential()
    model.add(Convolution2D(3, 1, 1,
                            border_mode='valid',
                            input_shape=(h,w,ch)))
    model.add(Activation('elu'))
    model.add(Convolution2D(32, 3, 3,border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3,border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('elu'))
    model.add(Convolution2D(128, 3, 3,border_mode='valid'))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(1024))
    model.add(Dropout(0.50))
    model.add(Activation('elu'))
    model.add(Dense(512))
    model.add(Dropout(0.50))
    model.add(Activation('elu'))
    model.add(Dense(128))
    model.add(Dropout(0.50))
    model.add(Activation('elu'))
    model.add(Dense(1))

    model.compile(loss = 'mse',
              optimizer='Adam')

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle Trainer') 
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--data_dir',type=str, default='data',help='Data directory')
    args = parser.parse_args()

    # Since the data in driving_log.csv (from simulator) does not have description headers, I make a header look-up 
    data_dict = {'center' : 0, 'left': 1, 'right':2,'steering':3,'throttle':4,'brake':5,'speed':6}
    t = time.time()
    X_train, y_train = load_data(args.data_dir,data_dict,0)
    X_udacity_data, y_udacity_data = load_data("data",data_dict,1)

    #Concatenate Udacity and self-collected data
    X_train = np.concatenate((X_train,X_udacity_data),axis=0)
    y_train = np.concatenate((y_train,y_udacity_data),axis=0)
    print("Time to load data = %.3f" % (time.time()-t))

    # Instantiate the model
    model = my_model3()

    # Train model
    history = model.fit(X_train,y_train,batch_size=args.batch,nb_epoch=args.epoch,validation_split = 0.2,shuffle=True)

    # Storing the model in another directory. This is to prevent overwriting the model files in the current running directory
    print("Saving model weights and configuration file.")
    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/model.h5", True)
    with open('./outputs/steering_model/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    
        
