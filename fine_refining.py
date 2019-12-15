# -*- coding: utf-8 -*-
"""
@author: Tanveer
"""

from __future__ import print_function

import numpy as np
import glob
import os
import caffe
import cv2
from scipy.spatial import distance

proto = "Models/deploy.prototxt"
model = "Models/squeezenet_v1.1.caffemodel"
caffe.set_mode_cpu()
net = caffe.Net(proto, model, caffe.TEST)

def extract_frame_features(frame):

    resized_image = cv2.resize(frame, (227,227))
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255)
    net.blobs['data'].reshape(1, 3, 227, 227)
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image)
    net.forward()
    features = net.blobs['pool10'].data[0].reshape(1,1000)
    single_featurevector = np.array(features)
    return single_featurevector
 

def fine_refine(seq1_features,seq2_features):

    mi = distance.euclidean(seq1_features,seq2_features)
    return mi

def directory_processing():
    frames_counter = -1
    seq_features = []
    previous_seq_features = np.zeros(5000)
    video_number = 'v3'
    path_for_video = video_number + "\\SqueezeNet\\Coarse-refine\\*.jpg"
    images_path = glob.glob(path_for_video)
    images_path = sorted(f for f in images_path)
    images_names = []
    #images_path.sort(key=lambda f: int(filter(str.isdigit, f)))

    for single_image in images_path:

        frames_counter = frames_counter + 1
        #image = cv2.imread(single_image)
        tinu = os.path.basename(single_image)
        tinu = os.path.splitext(tinu)[0]
        images_names.append(tinu)
    
    images_names = np.array(images_names,dtype=int)
    images_names = np.sort(images_names,axis=0)
    length = images_names.shape
    length = length[0]

    for index in range(length):
        image_name = images_names[index]
        full_name = video_number + "\\SqueezeNet\\Coarse-refine\\" + str(image_name) + '.jpg'
        image = cv2.imread(full_name)

        print ('Processing:', full_name)
        single_featurevector = extract_frame_features(image)
        

        seq_features.append(single_featurevector)

        if index%5 == 4:
            temp = np.asarray(seq_features)
            
            temp = temp.reshape(5000)
            features_distance = fine_refine(previous_seq_features,temp)
            previous_seq_features = temp
            
            seq_features = []
            print ('Distance ************************ = ', features_distance)

            if features_distance >= 7000: #7000 for squeezenet, 40000 for mobilenet, #49 for googlenet, #30 for alexnet
                name = video_number + '\\SqueezeNet\\Candidate-keyframes\\'+str(index)+'.jpg'
                
                cv2.imwrite(name,image)
                print ('Frame written ', name, ',distance = ',features_distance)
            #print ('m i = ' , features_distance)
            cv2.imshow('F',image)
            cv2.waitKey(1)

directory_processing()
 


















