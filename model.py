#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def atconv(net, rate, out_channels, kernel=3, relu=True):
    in_channels = net.get_shape().as_list()[3]
    filter_shape = np.asarray([kernel, kernel, in_channels, out_channels])
    filters = tf.Variable(tf.random_normal(filter_shape, stddev=0.3))
    net = tf.nn.atrous_conv2d(net, filters, rate, padding="SAME")
    if (relu):
        net = tf.nn.relu(net)
    else:
        net = tf.nn.sigmoid(net)
    return net

class Model():
    def __init__(self, batch_size=10, fsub=400, n_channels=3, f1=3, f2=1, f3=5, n1=64, n2=32, is_training=True):
        self.batch_size = batch_size
        self.fsub = fsub
        self.n_channels = n_channels
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.n1 = n1
        self.n2 = n2
        self.is_training = is_training

    def build_model(self):
        if self.is_training:
            self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.fsub, self.fsub, self.n_channels])
        else:
            self.input = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channels])

        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.fsub, self.fsub, 1])

        net = self.input
        net = atconv(net, 1, self.n_channels * 2)
        net = atconv(net, 1, self.n_channels * 2)
        net = atconv(net, 2, self.n_channels * 4)
        net = atconv(net, 2, self.n_channels * 8)
        net = atconv(net, 2, self.n_channels * 16)
        net = atconv(net, 2, self.n_channels * 32)
        net = atconv(net, 1, self.n_channels * 32)
        net = atconv(net, 1, 1, kernel=1, relu=False)
        self.output = net

        self.output_mag = tf.nn.l2_loss(self.output)
        self.label_mag = tf.nn.l2_loss(self.label)
        self.label_min = tf.reduce_min(self.label)
        self.label_max = tf.reduce_max(self.label)
        self.loss = tf.nn.l2_loss(self.output - self.label)
        # for reg_loss in tf.losses.get_regularization_losses():
        #     self.loss += reg_loss

if __name__ == "__main__":
    print("Testing model...")
    m = Model()
    m.build_model()
    check = tf.add_check_numerics_ops()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    images = np.random.uniform(size=[10, 33, 33, 3])
    labels = np.random.uniform(size=[10, 33, 33, 1])
    output, loss, check = sess.run([m.output, m.loss, check], feed_dict={m.input: images, m.label: labels})
    print("Input has shape:  {}".format(images.shape))
    print("Output has shape: {}".format(output.shape))
    print("Loss: {}".format(loss))
