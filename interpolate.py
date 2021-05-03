import argparse
import os
import sys
import glob
import logging
import cv2

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch

from src.archs import generator
from scipy.misc import imsave

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--data_path', type=str, help='path of faces', default='/home/ubuntu/arm/test/rawpred1_gaze_patch/r_eye')
parser.add_argument('--image_size', type=int, default=64,
                    help='size of cropped images')
parser.add_argument('--log_dir', type=str, help='path of eval checkpoint', default='/home/ubuntu/gaze_redirection/log')
parser.add_argument('--out_dir', type=str, help='path of eval checkpoint', default='/home/ubuntu/gaze_redirection/results_rawpred1_gaze_patch_r_eye')
parser.add_argument('--min_h_angle', type=int, help='minimum h angle', default=-10)
parser.add_argument('--max_h_angle', type=int, help='maximum h angle', default=10)
parser.add_argument('--min_v_angle', type=int, help='minimum v angle', default=-10)
parser.add_argument('--max_v_angle', type=int, help='maximum v angle', default=10)
parser.add_argument('--step_angle', type=int, help='step to sample angle', default=2)

params = parser.parse_args()

filelist = glob.glob(os.path.join(params.data_path, '*.png'))

if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)

class Model(object):
    def __init__(self, params):
        self.params = params

checkpoint = tf.train.latest_checkpoint(params.log_dir)

x_test_r = tf.placeholder(tf.float32, shape=(params.batch_size, params.image_size, params.image_size, 3))
angles_test_r = tf.placeholder(tf.float32, shape=(params.batch_size, 2))

x_fake = generator(x_test_r, angles_test_r)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
saver = tf.train.Saver()

with tf.Session(config=tf_config) as test_sess:
    with test_sess.graph.as_default():
        saver.restore(test_sess, checkpoint)

        for file in filelist:

            img_name = '.'.join(file.split('/')[-1].split('.')[:-1])
            print(img_name)

            np_x_test_r = Image.open(file)
            np_x_test_r = np_x_test_r.resize((params.image_size, params.image_size))
            np_x_test_r = np.array(np_x_test_r).astype(np.float32) / 127.5 - 1.0
            np_x_test_r = np.expand_dims(np_x_test_r, axis=0)


            video_name = os.path.join(params.out_dir, img_name + '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(
                video_name,
                fourcc,
                fps=25,
                frameSize=(params.image_size, params.image_size)
            )

            for h_angle in range(params.min_h_angle, params.max_h_angle, params.step_angle):
                for v_angle in range(params.min_v_angle, params.max_v_angle, params.step_angle):
                    np_angles_test_r = np.array([[h_angle, v_angle]])

                    _x_fake = test_sess.run(
                        [x_fake],
                        feed_dict={
                        x_test_r: np_x_test_r,
                        angles_test_r: np_angles_test_r}
                    )
                    imsave(
                        os.path.join(params.out_dir, img_name + '_H%d_V%d.png' % (h_angle, v_angle)), 
                        _x_fake[0][0]
                    )

                    frame_out = cv2.cvtColor(np.uint8((_x_fake[0][0] + 1.0) * 127.5), cv2.COLOR_BGR2RGB)

                    video.write(frame_out)

            cv2.destroyAllWindows()
            video.release()


