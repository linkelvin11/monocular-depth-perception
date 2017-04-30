#!/usr/bin/env python

import time
import tensorflow as tf

save_path = "./saved_model/model.ckpt"

class Trainer():
    def __init__(self, sess, model, fileReader):
        self.sess = sess
        self.model = model
        self.fileReader = fileReader
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(model.loss)

    def train_iter(self, batcher, validate=False, saver=None, path=None, val=None):
        batch = self.sess.run(batcher)
        input_batch = batch[0] 
        label_batch = batch[1]
        self.sess.run(self.optimizer, feed_dict={self.model.input: input_batch, self.model.label: label_batch})

    def train(self, saver=None, path=None, val=None):

        for i in range(100000000):
            train_batch = self.fileReader.get_batch()
            self.train_iter(train_batch, i % 10 == 0, saver=saver, path=path, val=val)
            if i % 100 == 0:
                val_batch = self.fileReader.get_val_batch()
                self.val(val_batch, saver=saver, path=path, iteration=i)

    def val(self, batcher, saver=None, path=None, val=None, iteration=0):
        batch = self.sess.run(batcher)
        input_batch = batch[0] 
        label_batch = batch[1]
        loss, mag, label_mag, label_min, label_max = self.sess.run([
            self.model.loss
            , self.model.output_mag
            , self.model.label_mag
            , self.model.label_min
            , self.model.label_max
        ], feed_dict={self.model.input: input_batch, self.model.label: label_batch})
        print('loss: {}\tmag: {}\tlabel_mag: {}\titeration: {}'.format(loss, mag, label_mag, iteration))
        if saver and path:
            sp = saver.save(self.sess, path)

