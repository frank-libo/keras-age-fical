# keras-age-fical
一个训练并且预测 年龄/表情的代码


用 train_datagen = ImageDataGenerator(
        # preprocessing_function=preprocess_input_new,  # ((x/255)-0.5)*2  归一化到±1之间
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
进行inceptionv3 ，EfficientNetB4 以及mini_XCEPTION 进行训练，无论取数据是直接train_datagen.flow_from_directory目录，
还是读取fer2013表格，效果均特别差。

用# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)

通过测试，图像增强，optimizer和 ir reduce 都会影响结果。
