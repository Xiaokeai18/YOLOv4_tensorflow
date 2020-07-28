# coding:utf-8
# test yolov4.weights

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config
from utils import tools
from src.YOLO import YOLO
import cv2
import numpy as np
import os
from os import path
import time

def seek_bound(bound,boxes,restriction):    #(640,360) , (left,right,top,bottom) , (1920,1080)
    left,right,top,bottom = boxes[0],boxes[1],boxes[2],boxes[3]
    width,height = bound[0],bound[1]
    RESHAPE = False
    if right-left>width:
        width = right-left
        RESHAPE = True
    if bottom-top>height:
        height = bottom-top
        RESHAPE = True
    if RESHAPE:
        if width*bound[1]>height*bound[0]:
            height = width*bound[1]//bound[0]
        else:
            width = height*bound[0]//bound[1]
    target = [(left+right-width)//2,(left+right+width)//2,(top+bottom-height)//2,(top+bottom+height)//2,RESHAPE]
    if target[0]<0:
        target[1] -= target[0]
        target[0] = 0
    if target[1]>restriction[0]:
        target[0] -= (target[1]-restriction[0])
        target[1] = restriction[0]
    if target[2]<0:
        target[3] -= target[2]
        target[2] = 0
    if target[3]>restriction[1]:
        target[2] -= (target[3]-restriction[1])
        target[3] = restriction[1]
    return target
    

def main():
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401        #608 anchors
    #anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    yolo = YOLO(80, anchors,width=416, height=416)

    SmallWindow = [(1280,0),(1280,360),(1280,720),(640,720),(0,720)]

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, isTrain=False)
    pre_boxes, pre_score, pre_label = yolo.get_predict_result(feature_y1, feature_y2, feature_y3, 80, 
                                                                                                score_thresh=config.val_score_thresh, iou_thresh=config.iou_thresh, max_box=config.max_box)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state("./yolo_weights")
        if ckpt and ckpt.model_checkpoint_path:
            print("restore: ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            exit(1)

        INPUT_SIZE = 416
        
        cap = cv2.VideoCapture(0)
        cap.set(6, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(3, 1920)
        cap.set(4, 1080)
        frameW = cap.get(3)
        frameH = cap.get(4)
        while True:

            start = time.perf_counter()
            
            output = np.zeros((1080,1920,3),np.uint8)
            targerboxs =[]

            _, frame = cap.read()
            img_rgb = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_in = img_rgb.reshape((1, INPUT_SIZE, INPUT_SIZE, 3)) / 255.

            boxes, scores, labels = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img_in})
            
            end = time.perf_counter()

            MainFrame = frame.copy()
            cv2.putText(MainFrame, 'infer time:%.2f' %((end-start)*1000), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 50, 50), 1)

            
            for i in range(len(scores)):
                if labels[i] != 0:
                    continue
                left = int(boxes[i][0] * frameW)
                right = int(boxes[i][2] * frameW)
                top = int(boxes[i][1] * frameH)
                bottom = int(boxes[i][3] * frameH)
            
                cv2.rectangle(MainFrame, (left, top), (right, bottom), (255,255,0), 2)
                cv2.putText(MainFrame, 'conf:%.2f' % scores[i], (left, top),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 250, 150), 2)
                
                targetl,targetr,targett,targetb,RESHAPE = seek_bound((640,360),(left,right,top,bottom),(1920,1080))
                
                targerboxs.append([targetl,targetr,targett,targetb,RESHAPE])
                
            for i in range(min(5,len(targerboxs))):
                S_left,S_top = SmallWindow[i]
                targerbox = targerboxs[i]
                print(targerbox)
                output[S_top:S_top+360,S_left:S_left+640,:] = cv2.resize(frame[targerbox[2]:targerbox[3],targerbox[0]:targerbox[1],:],(640,360))

            output[0:720,0:1280,:] = cv2.resize(MainFrame,(1280,720))
            cv2.imshow('img', output)
            if cv2.waitKey(1) & 0xFF == 27: 
                break


if __name__ == "__main__":
    main()
