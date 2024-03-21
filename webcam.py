"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model with a camera.
It will load a saved model from '--checkpoints_dir' and display the results in its own window.
It first creates model and dataset given the option. It will hard-code some parameters.
Example (You need to train models first or download pre-trained models from our website):

    Test a CycleGAN model (one side only):
        python3 webcam.py --name model2heart_latest --model test --preprocess none --no_dropout 
        
    If you are testing without CUDA ensure to set the gpu_ids option:
        python3 webcam.py --name model2heart_latest --model test --preprocess none --no_dropout --gpu_ids -1

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import cv2
import torch
import numpy as np
import time
import sys
sys.path.append('./util')  # Add the directory containing the util module to the Python path
from util import util  # Import the util module from the subdirectory

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    
    #start video/webcamsetup || Change index for different camera usage (e.g., index 1 for back facing camera if you have one)
    webcam = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not webcam.isOpened():
        raise IOError("Cannot open webcam")

    # Define variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    #the CycleGan takes data as a dictionary
    #easier to work within that constraint than to reright
    # start an infinite loop and keep reading frames from the webcam until we encounter a keyboard interrupt
    data = {"A": None, "A_paths": None, "B": None, "B_paths": None} # B and B_paths are needed for cycle_gan model
    while True:

        #ret is bool returned by cap.read() -> whether or not frame was captured succesfully
        #if captured correctly, store in frame
        ret, frame = webcam.read()

        #resize frame
        frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #model wants batchsize * channels * h * w
        #gives it a dimension for batch size
        frame = np.array([frame])
        #now shape is batchsize * channels * h * w
        frame = frame.transpose([0,3,1,2])

        #convert numpy array to tensor
        #need data to be a tensor for compatability with running model. expects floatTensors
        data['A'] = torch.FloatTensor(frame)
        model.set_input(data)  # unpack data from data loader
        model.test()

        #get only generated image - indexing dictionary for "fake" key
        result_image = model.get_current_visuals()['fake']
        #use tensor2im function provided by util file
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)  
        result_image = cv2.resize(result_image, (512, 512))      
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display FPS on the image
        cv2.putText(result_image, f'FPS: {fps:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('style', result_image)
        #ASCII value of Esc is 27
        c = cv2.waitKey(1)
        if c == 27: # ends on holding esc
            break   

    webcam.release()
    cv2.destroyAllWindows()