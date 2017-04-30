#!/usr/bin/env python

import time
import tensorflow as tf

save_path = "./saved_model/model.ckpt"

class Trainer():
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(model.loss)

    def train_iter(self, batcher, validate=False, saver=None, path=None, val=None):
        batch = self.sess.run(batcher)
        input_batch = batch[0] 
        label_batch = batch[1]
        self.sess.run(self.optimizer, feed_dict={self.model.input: input_batch, self.model.label: label_batch})

    def train(self, batcher, saver=None, path=None, val=None):
        if val is None:
            val = batcher
        for i in range(100000000):
            self.train_iter(batcher, i % 10 == 0, saver=saver, path=path, val=val)
            if i % 100 == 0:
                self.val(batcher, saver=saver, path=path, iteration=i)

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
        

if __name__ == '__main__':
    batch_size = 512
    import model
    from files import FileReader
    m = model.Model(batch_size=batch_size)
    m.build_model()
    with tf.Session() as sess:
        t = Trainer(sess, m)
        f = FileReader('./images/sets/train/*.JPEG', (33, 33), batch_size=batch_size)
        v = FileReader('./images/sets/validation/*.JPEG', (33, 33), batch_size=batch_size)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        try:
            saver.restore(sess,save_path)
        except:
            print('Error while restoring');
        f.start_queue_runners()
        v.start_queue_runners()
        t.train(f.get_batch(), val=v.get_batch())
        v.stop_queue_runners()
        f.stop_queue_runners()
