import os
import tensorflow as tf
import argparse
import numpy as np
import dataIO as io

def main():
    parser = argparse.ArgumentParser(description='py, data_list, num_per_tfrecord, outdir')

    parser.add_argument('--data_list', '-i1', default='F:/data_info/TFrecord/liver/set_5/down/64/alpha_0.1/fold_1/val.txt', help='data list')

    parser.add_argument('--num_per_tfrecord', '-i2', default=76, help='number per tfrecord')

    parser.add_argument('--outdir', '-i3', default='G:/data/tfrecord/liver/set_5/down/64/RBF/alpha_0.1/fold_1/val', help='outdir')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    data_set = io.load_matrix_data(args.data_list, 'float32')
    # data_set = io.load_matrix_raw_data(args.data_list, 'float')

    # shuffle
    # np.random.shuffle(data_set)
    print('data size: {}'.format(data_set.shape))

    num_per_tfrecord = int(args.num_per_tfrecord)
    num_of_total_image = data_set.shape[0]

    if (num_of_total_image % num_per_tfrecord != 0):
        num_of_recordfile = num_of_total_image // num_per_tfrecord + 1
    else:
        num_of_recordfile = num_of_total_image // num_per_tfrecord

    num_per_tfrecord_final = num_of_total_image - num_per_tfrecord * (num_of_recordfile - 1)

    print('number of total TFrecordfile: {}'.format(num_of_recordfile))

    # write TFrecord
    for i in range(num_of_recordfile):
        tfrecord_filename = os.path.join(args.outdir, 'recordfile_{}'.format(i + 1))
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        write = tf.python_io.TFRecordWriter(tfrecord_filename, options=options)

        print('Writing recordfile_{}'.format(i+1))

        if i == num_of_recordfile - 1:
            loop_buf = num_per_tfrecord_final
        else :
            loop_buf = num_per_tfrecord

        for image_index in range(loop_buf):
            image = data_set[image_index + i*num_per_tfrecord].flatten()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                }))

            write.write(example.SerializeToString())
        write.close()

if __name__ == '__main__':
    main()