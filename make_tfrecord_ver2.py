import os
import tensorflow as tf
import argparse
import numpy as np
import random
import dataIO as io

def main():
    parser = argparse.ArgumentParser(description='py, data_list, num_per_tfrecord, outdir')

    parser.add_argument('--data_list', '-i1', default='F:/data_info/TFrecord/liver/set_2/train.txt', help='data list')

    parser.add_argument('--num_per_tfrecord', '-i2', default=250, help='number per tfrecord')

    parser.add_argument('--outdir', '-i3', default='F:/data/tfrecord/liver/test', help='outdir')

    parser.add_argument('--tfrc_index', '-i4', default='1', help='tfrecord index')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # load list
    input_list = io.load_list(args.data_list)

    # shuffle
    random.shuffle(input_list)
    print('data size: {}'.format(len(input_list)))

    num_per_tfrecord = int(args.num_per_tfrecord)
    num_of_total_image = len(input_list)

    if (num_of_total_image % num_per_tfrecord != 0):
        num_of_recordfile = num_of_total_image // num_per_tfrecord + 1
    else:
        num_of_recordfile = num_of_total_image // num_per_tfrecord

    num_per_tfrecord_final = num_of_total_image - num_per_tfrecord * (num_of_recordfile - 1)

    print('number of total TFrecordfile: {}'.format(num_of_recordfile))

    # write TFrecord
    for i in range(num_of_recordfile):
        tfrecord_filename = os.path.join(args.outdir, 'recordfile_{}'.format(args.tfrc_index))
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        write = tf.python_io.TFRecordWriter(tfrecord_filename, options=options)

        print('Writing recordfile_{}'.format(i+1))

        if i == num_of_recordfile - 1:
            loop_buf = num_per_tfrecord_final
        else :
            loop_buf = num_per_tfrecord

        for image_index in range(loop_buf):
            # load data
            print('image from: {}'.format(input_list[image_index + i*num_per_tfrecord]))
            data = io.read_mhd_and_raw(input_list[image_index + i*num_per_tfrecord]).astype('float32')
            image = data.flatten()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                }))

            write.write(example.SerializeToString())
        write.close()

if __name__ == '__main__':
    main()