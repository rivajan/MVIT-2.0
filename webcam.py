"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/test_options.py for more test options.
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
import sys
sys.path.append('./util')  # Add the directory containing the util module to the Python path
from util import util  # Import the util module from the subdirectory

#text set up -- Can probably remove
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (0, 25) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 255, 255) 
# Line thickness of 2 px 
thickness = 2

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
    
    #start video/webcamsetup
    webcam = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not webcam.isOpened():
        raise IOError("Cannot open webcam")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (512, 512))

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
        #data['B'] = torch.FloatTensor(frame) # Needed for cycle_gan model
        model.set_input(data)  # unpack data from data loader
        model.test()

        #get only generated image - indexing dictionary for "fake" key
        result_image = model.get_current_visuals()['fake'] # 'fake_B' for cycle_gan model
        #use tensor2im function provided by util file
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)  
        result_image = cv2.resize(result_image, (512, 512))      
        #result_image = cv2.putText(result_image, str(opt.name)[6:-11], org, font,  # can probably remove
        #           fontScale, color, thickness, cv2.LINE_AA)  
        #out.write(result_image) 
        cv2.imshow('style', result_image)
        out.write(result_image)
        #ASCII value of Esc is 27. Can probably remove the c == 99 one
        c = cv2.waitKey(1)
        if c == 27: # ends on holding esc, maybe we can do something else, like when the window closes or something
            break   # maybe instead of displaying in a window here, can do the react thing, but might not be needed
        """
        if c == 99:
            if style_model_index == len(style_models): # not needed for us, but to remove error you could add 2 lines that he didn't add found on the webpage
                style_model_index = 0
            opt.name = style_models[style_model_index]
            style_model_index += 1
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt) 
        """

    webcam.release()
    cv2.destroyAllWindows()