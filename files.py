#!/usr/bin/env python

import tensorflow as tf
import random
import numpy as np

def create_input_pipeline_from_files_list(image_list, label_list, crop_shape, batch_size):
    filename_queue = tf.train.slice_input_producer([
        image_list
        , label_list
        ] , shuffle=True)

    # Decode image png
    image_file_content = tf.read_file(filename_queue[0])
    image = tf.image.decode_png(image_file_content, channels = 3)

    # Decode label png
    label_file_content = tf.read_file(filename_queue[1])
    label = tf.image.decode_png(label_file_content, channels = 1, dtype=tf.uint16)

    # Crop the image to the desired size
    image_crop_shape = tf.concat([crop_shape, [3]], 0)
    label_crop_shape = tf.concat([crop_shape, [1]], 0)
    rand_seed = random.random()*10000
    cropped_image = tf.random_crop(image, image_crop_shape, seed=rand_seed)
    cropped_image = tf.cast(cropped_image, tf.float32)/255.
    # Crop the label to the desired size
    if __name__ == "__main__":
        cropped_label = tf.random_crop(image, image_crop_shape, seed=rand_seed)
        cropped_label = tf.cast(cropped_label, tf.float32)/255.
    else:
        max_dist = 2000.
        cropped_label = tf.random_crop(label, label_crop_shape, seed=rand_seed)
        cropped_label = tf.cast(cropped_label, tf.float32)
        cropped_label = tf.clip_by_value(cropped_label, 0, max_dist)/max_dist


    # Create a batch
    image_batch, label_batch = tf.train.batch([
        cropped_image
        , cropped_label
        ], batch_size=batch_size)
    return [image_batch, label_batch]

class FileReader():
    """
    FileReader performs the cropping and batching for input files
    """

    def __init__(self, image_glob, label_glob, crop_shape, batch_size=10, dtype=tf.float32):
        """
        Creates a filereader which produces batches from a list of images and a list of labels
        """

        np.random.seed(seed=1234)
        partition = np.random.choice(3, len(image_glob), p=[0.98, 0.01, 0.01])
        np.random.seed()

        train_images, val_images, test_images = tf.dynamic_partition(image_glob, partition, 3)
        train_labels, val_labels, test_labels = tf.dynamic_partition(label_glob, partition, 3)

        print (val_labels.eval())

        self.batch = create_input_pipeline_from_files_list(train_images, train_labels, crop_shape, batch_size)

        self.val_batch = create_input_pipeline_from_files_list(val_images, val_labels, crop_shape, batch_size)

        self.test_batch = create_input_pipeline_from_files_list(test_images, test_labels, crop_shape, batch_size)

    def get_batch(self):
        """
        Returns a batch.
        """
        return self.batch

    def get_val_batch(self):
        """
        Returns a validation batch
        """
        return self.val_batch

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
