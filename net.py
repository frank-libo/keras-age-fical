import json
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers
from keras_applications_libo.efficientnet import EfficientNetB4
import keras

# create the base pre-trained model
def build_model(nb_classes, input_shape):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3),
    #                             backend=keras.backend,
    #                             layers=keras.layers,
    #                             models=keras.models,
    #                             utils=keras.utils)
# 注意：InceptionV3等自带的模型都先调用 C:\Python35\Lib\site-packages\keras\applications里面的Inception_v3.py
# 里面的函数，然后再调用C:\Python35\Lib\site-packages\keras_applications里面的Inception_v3.py，自己加的函数我放在了
# C:\Python35\Lib\site-packages\keras_applications_libo，直接调用。
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)

    # for layer in model.layers:
    #     print(layer.name)

    # for layer in model.layers[:17]:
    #     layer.trainable = False
    # for layer in model.layers[17:]:
    #     layer.trainable = True

    for layer in base_model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    print("starting model compile")
    return model


# model.save(model_name)优势和弊端：
# 优势一在于模型保存和加载就一行代码，写起来很方便。
# 优势二在于不仅保存了模型的结构和参数，也保存了训练配置等信息。以便于从上次训练中断的地方继续训练优化。
# 劣势就是占空间太大，我的模型用这种方式占了一个G。【红色部分就是上述模型采用第一种方式保存的文件】本地使用还好，如果是多人的模块需要集成，上传或者同步将会很耗时。

# model.to_json()转换成json，优势和弊端：
# 优势就是节省了硬盘空间，方便同步和协作
# 劣势是丢失了训练的一些配置信息
def save(model, tags, prefix):
    model.save_weights(prefix+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)


def load(prefix):
    # load json and create model
    with open(prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # model = build_model(5)
    # load weights into new model
    model.load_weights(prefix+".h5")
    # tags = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

