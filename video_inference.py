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


def main():
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401        #608 anchors
    #anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    yolo = YOLO(80, anchors,width=416, height=416)

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

        # id to names
        word_dict = tools.get_word_dict("./data/coco.names")
        # color of corresponding names
        color_table = tools.get_color_table(80)

        width = 416
        height = 416
        
        cap = cv2.VideoCapture(0)
        cap.set(6, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # cap.set(3, 1920)
        # cap.set(4, 1080)
        while True:

            start = time.perf_counter()
            
            _, frame = cap.read()
            img_rgb = cv2.resize(frame, (width, height))
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_in = img_rgb.reshape((1, width, height, 3)) / 255.
            #img, img_ori = read_img(img_name, width, height)

            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img_in})
            
            end = time.perf_counter()

            print("time:%f s" %(end-start))

            frame = tools.draw_img(frame, boxes, score, label, word_dict, color_table)

            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                break


if __name__ == "__main__":
    main()
