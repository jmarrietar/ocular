#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# Code from https://stackoverflow.com/questions/55598516/tensorflow-count-number-of-examples-in-a-tfrecord-file-without-using-depreca 

def count_tfrecord_examples(
        tfrecords_dir: str,
) -> int:
    """
    Counts the total number of examples in a collection of TFRecord files.

    :param tfrecords_dir: directory that is assumed to contain only TFRecord files
    :return: the total number of examples in the collection of TFRecord files
        found in the specified directory
    """

    count = 0
    for file_name in os.listdir(tfrecords_dir):
        print(file_name)
        tfrecord_path = os.path.join(tfrecords_dir, file_name)
        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path,'GZIP'))

    return count

print('{} Images'.format(count_tfrecord_examples('tf-records')))

