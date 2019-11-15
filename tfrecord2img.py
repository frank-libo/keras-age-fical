import os
import tensorflow as tf
from datetime import *
import cv2


def data_files(data_dir, subset):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if subset not in ['train', 'validation']:
        print('Invalid subset!')
        exit(-1)

    tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    print(data_files)
    if not data_files:
      print('No files found for data dir %s at %s' % (subset, data_dir))

      exit(-1)
    return data_files


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = 255* tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def parse_example_proto(example_serialized):
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),

      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1),

  }

  features = tf.parse_single_example(example_serialized, feature_map)
  # label = tf.cast(features['image/class/label'], dtype=tf.int32)
  return features['image/encoded'], features['image/class/label'], features['image/filename']




def batch_inputs(data_dir, image_size):
    outpath = './'
    files = data_files(data_dir, 'train')
    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=True,
                                                    capacity=16)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    ##################################
    # for temp in files:
    #     pp = os.path.join(data_dir, temp)
    #     for record in tf.python_io.tf_record_iterator(pp):
    #         image_buffer, label_index, fname = parse_example_proto(record)
    # 但是这个速度比下面这个速度差太远了，效果是一样的
    ######################################

    image_buffer, label_index, fname = parse_example_proto(serialized_example)
    # img = tf.decode_raw(image_buffer, tf.uint8)
    img = decode_jpeg(image_buffer)
    # img = tf.random_crop(img, [227, 227, 3])
    # img = tf.reshape(img, [image_size, image_size, 3])
    label = tf.cast(label_index, tf.int32)

    with tf.Session() as sess:
        tf.train.start_queue_runners()
        for i in range(0, 11823):
            outimg, outlabel = sess.run([img, label])
            outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
            path = os.path.join(outpath, str(outlabel[0]))
            if not os.path.exists(path):
                os.makedirs(path)
            nowTime = datetime.now().strftime("%Y%m%d%H%M%S%f")#生成当前的时间
            filename = os.path.join(path, str(nowTime))

            cv2.imwrite(filename + '.jpg', outimg)



batch_inputs('D:/Project/classify/age/rude-carnie-master/out/test_fold_is_0', 227)