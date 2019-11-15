# import tensorflow as tf
# import numpy as np
# import sys
# from train_models.mtcnn_model import P_Net as PNet
#
#
# # Edit just these
# FILE_PATH = 'D:\\Project\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\PNet_landmark\\'
# NUM_CLASSES = 26
# OUTPUT_FILE = 'alexnet_20171125_124517_epoch7.npy'
#
#
# if __name__ == '__main__':
#     saver = tf.train.import_meta_graph('D:\\Project\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\PNet_landmark\\PNet-18.meta')
#     layers = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4_1', 'conv4_2', 'conv4_3', 'cls_prob', 'bbox_pred', 'landmark_pred']
#     data = {
#         'conv1': [],
#         'pool1': [],
#         'conv2': [],
#         'conv3': [],
#         'conv4_1': [],
#         'conv4_2': [],
#         'conv4_3': [],
#         'cls_prob': [],
#         'bbox_pred': [],
#         'landmark_pred': []
#     }
#
#     with tf.Session() as sess:
#         # saver.restore(sess, FILE_PATH)
#         saver.restore(sess, tf.train.latest_checkpoint(FILE_PATH))
#
#         graph = tf.get_default_graph()
#
#         conv1 = graph.get_tensor_by_name('conv1:0')
#         # w = graph.get_operation_by_name("word_embedding/conv1").outputs[0]
#         # print(w)
#         for op_name in layers:
#             with tf.variable_scope(op_name, reuse = True):
#                 biases_variable = tf.get_variable('net')
#             #     # weights_variable = tf.get_variable('weights')
#                 data[op_name].append(sess.run(biases_variable))
#                 # data[op_name].append(sess.run(weights_variable))
#
#         # np.save(OUTPUT_FILE, data)





import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path = 'D:\\Project\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\PNet_landmark\\PNet-18.index'  # your ckpt path
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

alexnet = {}
alexnet_layer = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4_1', 'conv4_2', 'conv4_3', 'cls_prob', 'bbox_pred', 'landmark_pred']
add_info = ['weights', 'biases']

alexnet = {'conv1': [],
        'pool1': [],
        'conv2': [],
        'conv3': [],
        'conv4_1': [],
        'conv4_2': [],
        'conv4_3': [],
        'cls_prob': [],
        'bbox_pred': [],
        'landmark_pred': []}

for key in alexnet_layer:
    # print ("tensor_name",key)

    str_name = key
    # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    if str_name.find('Adam') > -1:
        continue

    print('tensor_name:', str_name)

    if str_name.find('/') > -1:
        names = str_name.split('/')
        # first layer name and weight, bias
        layer_name = names[0]
        layer_add_info = names[1]
    else:
        layer_name = str_name
        layer_add_info = None

    if layer_add_info == 'weights':
        alexnet[layer_name][0] = reader.get_tensor(key)
    elif layer_add_info == 'biases':
        alexnet[layer_name][1] = reader.get_tensor(key)
    else:
        alexnet[layer_name] = reader.get_tensor(key)

# save npy
np.save('alexnet_pointing04.npy', alexnet)
print('save npy over...')
# print(alexnet['conv1'][0].shape)
# print(alexnet['conv1'][1].shape)

