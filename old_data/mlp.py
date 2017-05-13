from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data

import scipy.io
import numpy as np
import sklearn as sk
import sklearn.utils as sku
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep

import tensorflow as tf
import data_method as dat
import copy 
import matplotlib.pyplot as plt

FLAGS = None



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # import data from the file
    whole_data, whole_label = dat.getData()
    print(whole_data.shape)
    print(whole_label.shape)
    data,testData,label, testLabel = train_test_split(whole_data,whole_label, test_size = 0.3, random_state = 1)


    n_hidden_1 = 25 # 1st layer number of features
    n_hidden_2 = 10 # 2nd layer number of features
    n_input = data.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = label.shape[1] # MNIST total classes (0-9 digits)
    print (data.shape)
    print (label.shape)
    # First layer (3 value * 19 channels)
    x =  tf.placeholder(tf.float32, [None, n_input])
    w_0 = weight_variable([n_input, n_hidden_1])

    b_0 = bias_variable([n_hidden_1]) 
    # Node 0
    n_0 = tf.nn.relu(tf.matmul(x,w_0) + b_0)

    w_1 = weight_variable([n_hidden_1, n_hidden_2])

    b_1 = bias_variable([n_hidden_2])

    n_1 = tf.nn.relu(tf.matmul(n_0,w_1) + b_1)

    w_2 = weight_variable([n_hidden_2,n_classes])
    b_2 = bias_variable([n_classes])
    y = tf.matmul(n_1, w_2) + b_2

    # There are three outputs
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    train = []
    test  = []
    for i in range(100000):
        batch_xs, batch_ys = dat.shuffleData(data,label)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracy = accuracy.eval(feed_dict={
                x:batch_xs, y_: batch_ys})
        if i % 500 == 0:
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train += [train_accuracy]
            print(sess.run(accuracy, feed_dict={x:testData,y_:testLabel}))
            test += [sess.run(accuracy, feed_dict={x:testData,y_:testLabel})]



    print(sess.run(accuracy, feed_dict={x:testData,y_:testLabel}))
    # print(sess.run(y,feed_dict={x:batch_xs,y_:batch_ys}))
    # print(sess.run(y_,feed_dict={x:batch_xs,y_:batch_ys} ))
    t = np.linspace(1,len(train)*500,len(train),endpoint = True)
    print(t)
    print(train)
    print(test)
    plt.plot(t,train,'b--',t,test,'r--')
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


