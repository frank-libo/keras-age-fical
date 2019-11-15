import net
import gc
# import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from config import *
from keras.preprocessing.image import *

# calculate the accuracy on validation set
def myacc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)


nb_epochs = 50


def ir_Schedule(epoch):
    # Learning Rate Schedule
    lr = 0.0001

    check_1 = int(nb_epochs * 0.9)
    check_2 = int(nb_epochs * 0.8)
    check_3 = int(nb_epochs * 0.5)
    check_4 = int(nb_epochs * 0.3)

    if epoch > check_1:
        lr = 0.000001
    elif epoch > check_2:
        lr = 0.00000425
    elif epoch > check_3:
        lr = 0.00000625
    elif epoch > check_4:
        lr = 0.000025

    return lr

image_size = 227
def read_file_all(data_dir_path):
    img = []
    num = 0
    for f in os.listdir(data_dir_path):
        image_path = os.path.join(data_dir_path, f)
        if os.path.isfile(image_path):
            num = num + 1
            images = image.load_img(image_path, target_size=(image_size, image_size))
            x = image.img_to_array(images)
            x = np.expand_dims(x, axis=0)
            img.append(x)
    x = np.concatenate([x for x in img])

    return x, num


from keras.callbacks import Callback


class TestModel(Callback):
    def __init__(self, model, path, train_generator):
        self.model = model
        self.path = path
        self.labels = train_generator.class_indices
        self.sum_right = 0
        self.sum_num = 0

    def on_train_begin(self, logs=None):
        doc = open('./result.txt', 'a')
        print('*' * 40, file=doc)
        for f in os.listdir(self.path):
            image_path = os.path.join(self.path, f)
            testimage, num = read_file_all(image_path)
            pred = self.model.predict(testimage)
            predicted_class_indices = np.argmax(pred, axis=1)

            label = dict((v, k) for k, v in self.labels.items())

            # 建立代码标签与真实标签的关系
            predictions = [label[i] for i in predicted_class_indices]

            right = 0
            wrong = 0

            for prelabel in predictions:
                if prelabel == f:
                    right = right + 1
                else:
                    wrong = wrong + 1

            self.sum_right = self.sum_right + right
            self.sum_num = self.sum_num + num
            Tacc = right / num
            data_dict = "the acc of (%s) is %f" % (f, Tacc)
            print(data_dict, file=doc)

        all_acc = self.sum_right / self.sum_num
        data_dict = "all the acc is %f" % (all_acc)
        print(data_dict, file=doc)
        print('*' * 40, file=doc)
        doc.close()

    def on_epoch_end(self, epoch, logs={}):
        doc = open('./result.txt', 'a')
        print('*' * 40, file=doc)
        for f in os.listdir(self.path):
            image_path = os.path.join(self.path, f)
            testimage, num = read_file_all(image_path)
            pred = self.model.predict(testimage)
            predicted_class_indices = np.argmax(pred, axis=1)

            label = dict((v, k) for k, v in self.labels.items())

            # 建立代码标签与真实标签的关系
            predictions = [label[i] for i in predicted_class_indices]

            right = 0
            wrong = 0

            for prelabel in predictions:
                if prelabel == f:
                    right = right + 1
                else:
                    wrong = wrong + 1
            self.sum_right = self.sum_right + right
            self.sum_num = self.sum_num + num
            Tacc = right / num
            data_dict = "eopch:%d, the acc of (%s) is %f" % (epoch, f, Tacc)
            print(data_dict, file=doc)

        all_acc = self.sum_right / self.sum_num
        data_dict = "eopch:%d,all the acc is %f" % (epoch, all_acc)
        print(data_dict, file=doc)
        print('*' * 40, file=doc)
        doc.close()


import os
from models.cnn import mini_XCEPTION

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def train():
    INPUT_DATA = 'D:/Project/classify/demo/dataset/age/mtcnn_hunhe_train/'
    test_DATA = 'D:/Project/classify/demo/dataset/age/mtcnn_hunhe_train/'
    batch_size = 32
    # initial_learning_rate = 1e-4

    train_datagen = ImageDataGenerator(
        # preprocessing_function=preprocess_input_new,  # ((x/255)-0.5)*2  归一化到±1之间
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
    )


    train_generator = train_datagen.flow_from_directory(directory=INPUT_DATA,
                                                        target_size=(image_size, image_size),  # Inception V3规定大小
                                                        batch_size=batch_size)
    print(train_generator.class_indices)

    val_generator = train_datagen.flow_from_directory(directory=test_DATA,
                                                       target_size=(image_size, image_size),  # Inception V3规定大小
                                                       batch_size=batch_size)
    model = net.build_model(8, (image_size, image_size, 3))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[myacc])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # metrics: 评价函数, 与损失函数类似, 只不过评价函数的结果不会用于训练过程中, 可以传递已有的评价函数名称, 或者传递一个自定义的theano / tensorflow函数来使用, 自带的评价函数有: binary_accuracy(
    #     y_true, y_pred), categorical_accuracy(y_true, y_pred), sparse_categorical_accuracy(y_true,
    #     y_pred), top_k_categorical_accuracy(y_true, y_pred, k=5).自定义评价函数应该在编译的时候compile传递进去,
    # 该函数需要以(y_true, y_pred) 作为输入参数, 并返回一个张量作为输出结果.


    try:
        os.mkdir("./model/")
    except OSError:
        print(OSError)
        pass
    checkpoint = ModelCheckpoint(filepath='./model/weights.{epoch:02d}.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=False,
                                 period=1)

    path = 'D:/Project/classify/demo/dataset/age/mtcnn_hunhe_test/'
    testmodel = TestModel(model, path, train_generator)

    # lr_scheduler = LearningRateScheduler(ir_Schedule)
    # optimizer = Adam(lr=initial_learning_rate, decay=1e-6)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc])

    patience = 50
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    # 可以自己定义compile函数中传入的myacc，也可以用系统定义的 acc,lr,loss,val_loss,val_acc，来作为检测项目。

    callbacks = [checkpoint, reduce_lr, testmodel]
    model.fit_generator(generator=train_generator,
                                     steps_per_epoch=int(train_generator.samples / batch_size), #参数steps_per_epoch是通过把训练图像的数量除以批次大小得出的。
                                     epochs=50,          # 例如，有100张图像且批次大小为50，则steps_per_epoch值为2。参数epoch决定网络中所有图像的训练次数。
                                     validation_data=val_generator,
                                     validation_steps=1,
                                     class_weight='auto', callbacks=callbacks)


    model.save('./models/%s.h5' % model_name)
    del train_generator
    del val_generator
    del model
    gc.collect()


if __name__ == "__main__":
    train()

