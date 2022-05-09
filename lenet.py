import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
import numpy as np

class Lenet:
    def __init__(self):
        #self.raw_input_image = tf.placeholder(tf.float32, [None, 784])
        self.input_images = tf.placeholder(tf.float32, [None, 288, 180, 3])
        self.raw_input_label = tf.placeholder("float", [None, 436])
        self.input_labels = tf.cast(self.raw_input_label,tf.int32)
        self.dropout = cfg.KEEP_PROB

        with tf.variable_scope("Lenet") as scope:
            self.train_digits1, self.train_digits = self.construct_net(True)
            scope.reuse_variables()
            self.pred_digits1, self.pred_digits = self.construct_net(False)

        self.prediction = tf.argmax(self.pred_digits, 1)
        #self.prediction_max_prob=self.pred_digits[self.prediction]
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits1, 1), tf.argmax(self.input_labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


        self.loss1 = slim.losses.softmax_cross_entropy(self.train_digits1, self.input_labels)
        self.loss2 = slim.losses.softmax_cross_entropy(self.train_digits, self.input_labels)        
        self.lr = cfg.LEARNING_RATE
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(tf.add(self.loss1,self.loss2))


    def construct_net(self,is_trained = True):
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input_images,64,[27,27],1,padding='SAME',scope='conv1')
            #net = tf.nn.relu(net, name="relu1")
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net,128,[5,5],1,scope='conv3')
            #net = tf.nn.relu(net, name="relu2")
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net,256,[5,5],1,scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net,128,[3,3],1,scope='conv6')

            #net = tf.nn.relu(net, name="relu3")
            conv2d_results=net 

            conv2d_results = slim.flatten(conv2d_results, scope='flat9')
            conv2d_results = slim.fully_connected(conv2d_results, 256, scope='fc10') 
            conv2d_results = slim.dropout(conv2d_results, self.dropout,is_training=is_trained, scope='dropout11')
            digits1 = slim.fully_connected(conv2d_results, 436, scope='fc12')

            net=slim.avg_pool2d(net, [10,11],[10,11],scope='pool13')            
            net = slim.flatten(net, scope='flat14')
            net = slim.fully_connected(net, 256, scope='fc15')
            net = slim.dropout(net, self.dropout,is_training=is_trained, scope='dropout16')
            digits2 = slim.fully_connected(net, 436, scope='fc17')
        return digits1, digits2
