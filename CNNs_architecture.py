def deep_network(self):  # this function is used to encapsulate the CNN model in the time distributed layer, and append
    # the LSTM and Dense layers to the TDL layer
    if self.CNNModel == 1:  # based on the model chosen during the class initialization, the proper CNN will be selected
        featuresExt = CNN1(self.inputSize[1:])
    elif self.CNNModel == 2:
        featuresExt = CNN2(self.inputSize[1:])
    else:
        featuresExt = CNN3(self.inputSize[1:])
    input_shape = tf.keras.layers.Input(self.inputSize)
    TD = tf.keras.layers.TimeDistributed(featuresExt)(input_shape)  # encapsulating the CNN model in the TDL layer
    RNN = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(128))(TD)  # adding the LSTM layer
    Dense1 = tf.keras.layers.Dense(64, activation='relu')(RNN)  # adding the Dense layer
    Dense2 = tf.keras.layers.Dense(5, activation='softmax')(Dense1)  # last layer performs the classification of the input
    model_ = tf.keras.models.Model(inputs=input_shape, outputs=Dense2)
    return model_


def CNN1(shape):  # Lightest model: 5 blocks of convolutional blocks with batch normalization and average pooling layers.
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model


def CNN2(shape):  # Default model: 6 blocks of convolutional blocks with batch normalization and average pooling layers.
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x2)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model


def CNN3(shape):  # Heaviest model: 7 blocks of convolutional blocks with batch normalization and average pooling layers.
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x2)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x3 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x3)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x3])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model
