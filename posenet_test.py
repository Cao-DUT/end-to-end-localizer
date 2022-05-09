# Import the converted model's class
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import random
import tensorflow as tf
import cv2
from tqdm import tqdm
import time

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet
from Inference import inference


batch_size = cfg.BATCH_SIZE
# Set this path to your dataset directory
path='/home/freedom/LeNet/NCLT/LeNet-master-easy/checkpoint/'
directory = "CaoCollege/"
dataset = "8-13segment_pose1.txt"

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
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split(",") #
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
            pose_x = [1 if i == pose_x else 0 for i in range(740)]  # 1-hot result for tiger
            pose_x_batch.append(pose_x)
           
        yield np.array(image_batch), np.array(pose_x_batch)


def main():
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()

    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER

    images = tf.placeholder(tf.float32, [batch_size, 288, 180, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 740])
    datasource = get_data()

    saver = tf.train.Saver()

    saver.restore(sess, path+"variable.ckpt")
    print(path+"variable.ckpt")

    data_gen = gen_data_batch(datasource) 
    with open('PR_stats_0516.txt','w') as f:
        for i in range(len(datasource.images)):
        #batch = mnist.train.next_batch(50)
              
            np_image = datasource.images[i]

            #predition = sess.run(lenet.prediction, feed_dict={lenet.input_images: [np_image]})
            start_time=time.time()
            prob = sess.run(lenet.pred_digits1, feed_dict={lenet.input_images: [np_image]})
            end_time=time.time()
            with open('timing_0516.txt','a') as ti:
                ti.write(str(end_time-start_time)+'\n') 
                ti.close()           
            print(prob.shape)
            predition=tf.argmax(prob, 1)
            pred = np.argsort(-prob)
            #print(pred)
            top5 = [(pred[0][i], prob[0][pred[0][i]]) for i in range(5)]
            print(("Top5: ", top5))
            #predition=tf.constant(predition)
            #prob_prediction=prob(predition)
            #prob=lenet.pred_digits[predition]
            #print(str(i),"-->",str(predition),"-->",str(prob_prediction))
            #a_string=str(i)+' '+str(predition[0])+' '+str(prob_prediction)
            a_string=str(i)+' '+str(top5)
            f.write(a_string+'\n')
	#df=pd.DataFrame(prob)
	#df.to_csv("./predicted_prob.txt",mode='a')
        #train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
                  #lenet.input_images: np_image,lenet.raw_input_label: np_poses_x })

        #print("step %d, training accuracy %g" % (i, train_accuracy))
        #sess.run(lenet.train_op,feed_dict={lenet.input_images: np_images,lenet.raw_input_label: np_poses_x})
        #if i % 1000 == 0:
            #save_path = saver.save(sess, parameter_path)
    #save_path = saver.save(sess, parameter_path)




if __name__ == '__main__':
	main()
