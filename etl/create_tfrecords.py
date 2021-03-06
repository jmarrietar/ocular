from tfr_image import TFRimage

tool = TFRimage()
tool.create_tfrecords(
    dataset_dir="../data/sample/train",
    tfrecord_filename="ocular-labeled",
    num_shards=2,
)