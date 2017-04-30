#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    import sys
    import tensorflow as tf
    from model import Model

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, help='operating mode (train or generate)')
    parser.add_argument('--model', type=str, default='./saved_model/model.ckpt')
    parser.add_argument('--input', type=str, nargs='+', default=['./input/input.png'])
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--subimage-size', type=int, default=33)

    args = parser.parse_args()

    def normalize_for_output(tensor):
        tensor = tf.div(
            tf.subtract(
                tensor
                , tf.reduce_min(tensor)
            )
            , tf.subtract(
                tf.reduce_max(tensor)
                , tf.reduce_min(tensor)
            )
        )

        return tensor


    def reshape_for_output(image):
        shape = image.shape
        if len(shape) > 3:
            shape = shape[1:]
        return image.reshape(shape)

    def write_image_to_file(image, filename):
        image = reshape_for_output(image)*255.
        print (image.shape)
        images_out = tf.image.encode_png(image)
        fh = open(filename, "wb+")
        fh.write(images_out.eval())
        fh.close()


    if args.mode == 'train':
        from train import Trainer
        from files import FileReader
        import glob
        import re
        import os.path
        m = Model(batch_size=args.batch_size)
        m.build_model()
        with tf.Session() as sess:
            
            batch_size = args.batch_size
            crop_shape = (400,400)
            
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
            
            f = FileReader(
                filtered_image_glob
                , filtered_label_glob
                , crop_shape, batch_size=batch_size)
            
            t = Trainer(sess, m, f)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            try:
                saver.restore(sess, args.model)
            except:
                print('No save file found.  Creating new file at {}'.format(args.model));
            f.start_queue_runners()
            t.train(saver=saver, path=args.model)
            f.stop_queue_runners()
    elif args.mode == 'generate':
        if args.input is None:
            print("must provide an input file in generate mode")
            sys.exit(1)
        from files import FileReader
        import random
        from datetime import datetime
        m = Model(is_training=False)
        m.build_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            try:
                saver.restore(sess, args.model)
            except:
                print('Could not load model file: {}!'.format(args.model))
                sys.exit(1)
            # Generate new images here
            filename_queue = tf.train.string_input_producer(args.input)
            image_reader = tf.WholeFileReader()
            _, image_file = image_reader.read(filename_queue)
            image = tf.image.decode_png(image_file, channels=3)
            labelname_queue = tf.train.string_input_producer(['./input/label.png'])
            _, label_file = image_reader.read(labelname_queue)
            label = tf.image.decode_png(label_file, channels=1, dtype=tf.uint16)

            # Crop the image to the desired size
            random.seed(datetime.now())
            rand_seed = random.random()*1000
            print ('rand_seed: {}'.format(rand_seed))
            crop_shape = tf.concat([(400,400), [3]], 0)
            cropped = tf.random_crop(image, crop_shape, seed=rand_seed)
            cropped = tf.cast(cropped, tf.float32)/255.

            label_crop_shape = tf.concat([(400,400), [1]], 0)
            cropped_label = tf.random_crop(label, label_crop_shape, seed=rand_seed)
            cropped_label = tf.cast(cropped_label, tf.float32)
            max_dist = 2000.
            cropped_label = tf.clip_by_value(cropped_label, 0, max_dist)/max_dist
            
            cl = normalize_for_output(cropped_label)

            batch = tf.train.batch([cropped], batch_size=1)
            scale = 1

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            output, orig = sess.run(
                    [
                        normalize_for_output(m.output)*scale
                        , normalize_for_output(m.input)*scale
                    ]
                    ,  feed_dict={m.input: batch.eval()}
                )
            

            coord.request_stop()
            coord.join(threads)
            write_image_to_file
            write_image_to_file(cl.eval()*scale, './outputs/label.png')
            write_image_to_file(orig, './outputs/orig.png')

            # print(cl.eval()*255)
            # print(output*255)
            print(tf.reduce_max(cropped_label).eval())
            print(tf.reduce_min(cropped_label).eval())
            write_image_to_file(output, './outputs/output.png')
    else:
        print('Invalid "mode": {}!'.format(args.mode))
    sys.exit(0)
