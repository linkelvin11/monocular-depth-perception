#!/usr/bin/env python

import tensorflow as tf
import random


class FileReader():
    """
    FileReader manages all the queueing involved in loading images from files, taking random crops,
    and forming batches.
    """

    def __init__(self, image_glob, label_glob, crop_shape, batch_size=10, dtype=tf.float32):
        """
        Creates a FileReader that matches image filenames with `glob`,
        crops them to `crop_shape`, and produces batches of size `batch_size`.
        """
        # Create a queue of all the filenames
        # print ('matching filenames')
        # all_image_filenames = tf.train.match_filenames_once(image_glob)
        # all_label_filenames = tf.train.match_filenames_once(label_glob)
        # self.batch = [all_image_filenames, all_label_filenames]
        # return
        # print ('done matching filenames')

        filename_queue = tf.train.slice_input_producer([
            image_glob
            , label_glob
            ] , shuffle=True)

        # Decode image png
        image_file_content = tf.read_file(filename_queue[0])
        image = tf.image.decode_png(image_file_content, channels = 3)

        # Decode label png
        label_file_content = tf.read_file(filename_queue[1])
        label = tf.image.decode_png(label_file_content, channels = 1, dtype=tf.uint16)

        # Create a reader and decode a jpeg image
        # image_reader = tf.WholeFileReader()
        # _, image_file = image_reader.read(filename_queue)
        # image = tf.image.decode_jpeg(image_file, channels=3)

        # Crop the image to the desired size
        image_crop_shape = tf.concat([crop_shape, [3]], 0)
        label_crop_shape = tf.concat([crop_shape, [1]], 0)
        rand_seed = random.random()*10000
        cropped_image = tf.random_crop(image, image_crop_shape, seed=rand_seed)
        cropped_image = tf.cast(cropped_image, dtype)/255.
        # Crop the label to the desired size
        if __name__ == "__main__":
            cropped_label = tf.random_crop(image, image_crop_shape, seed=rand_seed)
            cropped_label = tf.cast(cropped_label, dtype)/255.
        else:
            max_dist = 2000.
            cropped_label = tf.random_crop(label, label_crop_shape, seed=rand_seed)
            cropped_label = tf.cast(cropped_label, dtype)
            cropped_label = tf.clip_by_value(cropped_label, 0, max_dist)/max_dist


        # Create a batch
        image_batch, label_batch = tf.train.batch([
            cropped_image
            , cropped_label
            ], batch_size=batch_size)
        self.batch = [image_batch, label_batch]

    def get_batch(self):
        """
        Returns a batch.
        """
        return self.batch

    def start_queue_runners(self):
        """
        Starts up the queue runner threads.  Must be called before executing.
        """
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def stop_queue_runners(self):
        """
        Stops the queue runner threads.  Should be called when finished executing.
        """
        self.coord.request_stop()
        self.coord.join(self.threads)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    import glob
    import os.path
    import re
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
        # Display M x N images, for a total batch size of M*N
        M = 5
        N = 2 
        batch_size = M*N
        crop_shape = (320, 240)

        # Open the file reader and generate 1 batch
        image_glob = glob.glob('./data/rgbd-dataset/*/*/*[0-9].png')
        image_glob.sort()
        label_glob = glob.glob('./data/rgbd-dataset/*/*/*[0-9]_depth.png')
        filtered_image_glob = []
        filtered_label_glob = []
        for filename in image_glob:
            labelname = re.sub('\.png$', '_depth.png', filename)
            if os.path.exists(labelname):
                filtered_image_glob.append(filename)
                filtered_label_glob.append(labelname)
        label_glob.sort()
        print (len(filtered_image_glob))
        print (len(filtered_label_glob))
        f = FileReader(
            filtered_image_glob
            , filtered_label_glob
            , crop_shape, batch_size=batch_size)
        tf.global_variables_initializer().run()
        f.start_queue_runners()
        batch = sess.run(f.get_batch())
        f.stop_queue_runners()

        # Plot the batch of images
        print ('plot batch images')
        fig = plt.figure()
        for m in range(M):
            for n in range(N):
                for o in range(2):
                    batch_index = m*N + n
                    plot_index = m*2*N + 2*n
                    axes = fig.add_subplot(M, 2 * N, plot_index + 1 + o)
                    axes.set_axis_off()
                    axes.imshow(batch[o][batch_index]*255, interpolation='nearest')
        fig.suptitle('Batch of {} sub-images of shape {}'.format(batch_size, crop_shape), fontsize=20)
        plt.show()
