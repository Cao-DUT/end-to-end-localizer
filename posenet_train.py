# Import the converted model's class
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import random
import tensorflow as tf
import cv2
from tqdm import tqdm
import pandas as pd
import time

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet


batch_size = cfg.BATCH_SIZE
# Set this path to your dataset directory
directory = "CaoCollege/"
dataset = "training.txt"

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = int ((new_height - output_side_length) / 2)
	width_offset = int ((new_width - output_side_length) / 2)
	cropped_img = img[height_offset:height_offset + output_side_length,
								 width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		#X = cv2.cvtColor(X,cv2.COLOR_RGB2GRAY)

		#X = cv2.resize(X, (224, 224))
		#X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	#N = 0
	#mean = np.zeros((1, 3, 288, 180))
	#for X in tqdm(images_cropped):
		        
		        #X = np.transpose(X,(2,0,1))
		        
		        #mean[0][0]+= X[:,:,0]
		        #mean[0][1]+= X[:,:,1]
		        #mean[0][2]+= X[:,:,2]
		        #N += 1
	#mean[0]/= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		#X = np.transpose(X,(2,0,1))
		#X = X 
		#X = np.squeeze(X)
		#X = np.transpose(X, (1,2,0))
		images_out.append(X)
	return images_out

def get_data():
	poses = []
	images = []

	with open(directory+dataset) as f:
		#print(f)
		#next(f)  # skip the 3 header lines
		#next(f)
		#next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6= line.split(",")  #,p1,p2,p3,p4,p5,p6
			p0 = float(p0)

			poses.append((p0))
			images.append(directory+fname)
	images = preprocess(images)
	return datasource(images, poses)

def gen_data(source):
	while True:
		indices = list(range(len(source.images)))
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i]
			
			yield image, pose_x

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        for _ in range(batch_size):
            image, pose_x = next(data_gen)
            #print(image.shape)
            #image=image[:,:,0]
 
            #print(image.shape)
            image_batch.append(image)
            pose_x = [1 if i == pose_x else 0 for i in range(436)]  # 1-hot result for tiger
            pose_x_batch.append(pose_x)
           
        yield np.array(image_batch), np.array(pose_x_batch)


def main():
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()

    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER

    images = tf.placeholder(tf.float32, [batch_size, 288, 180, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 436])
    datasource = get_data()

    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.global_variables_initializer())
    cao_accuracy=[]

   
    for i in range(max_iter):
        #batch = mnist.train.next_batch(50)
        data_gen = gen_data_batch(datasource)        
        np_images, np_poses_x = next(data_gen)
        train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
               lenet.input_images: np_images,lenet.raw_input_label: np_poses_x })
        start_time=time.time()
        sess.run(lenet.train_op,feed_dict={lenet.input_images: np_images,lenet.raw_input_label: np_poses_x})
        end_time=time.time()
        with open("./timing.txt",'a') as t:
              t.write(str(end_time-start_time))
              t.write('\n')
              t.close()

        with open("./training_accuracy.txt",'a') as f:
              f.write(str(train_accuracy))
              f.write('\n')
              f.close()
        if i % 100 == 0:
            
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 500 == 0:
            save_path = saver.save(sess, parameter_path)
    save_path = saver.save(sess, parameter_path)




if __name__ == '__main__':
	main()
