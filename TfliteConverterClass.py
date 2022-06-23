import shutil
import cv2.cv2 as cv2
import glob
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras_video import VideoFrameGenerator



def deep_network(self):  # this function is used to encapsulate the CNN model in the time distributed layer, and append
    # the LSTM and Dense layers to the TDL layer
    if self.CNNModel == 1:  # based on the model chosen during the class initialization, the proper CNN will be selected
        featuresExt = CNN1(self.inputSize[1:])
    elif self.CNNModel == 2:
        featuresExt = CNN2(self.inputSize[1:])
    else:
        featuresExt = CNN3(self.inputSize[1:])
    input_shape = tf.keras.layers.Input(shape=self.inputSize, batch_size=1)
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


def save_temporary_model(self, m):  # function used to create a pb file of the model to be converted
    neomodel = deep_network(self)
    neomodel.load_weights(m)
    neomodel.summary()

    # create the concrete function for the conversion of the h5 model in pb format
    run_model = tf.function(lambda x: neomodel(x))
    shape = np.asarray([neomodel.inputs[0].shape[i] for i in range(1, len(neomodel.inputs[0].shape))])
    shape = np.concatenate(([1], shape))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec(shape, neomodel.inputs[0].dtype))
    modelDirTmp = os.path.join(self.modelDirTmp, (m.split('/')[-1]).split('.')[0])
    if os.path.exists(modelDirTmp):
        shutil.rmtree(modelDirTmp)
        os.mkdir(modelDirTmp)
    else:
        os.mkdir(modelDirTmp)
    # convert the model in pb format
    neomodel.save(modelDirTmp, save_format="tf", signatures=concrete_func)
    print("Tmp folder for Model {} conversion created!".format(m.split('/')[-1]))


def representative_dataset_gen_Canny():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_Canny/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=1,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(250):
        yield [input_value]


def representative_dataset_gen_Roberts():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_Roberts/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=1,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(100):
        yield [input_value]


def representative_dataset_gen_Binary():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_Binary/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=1,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(100):
        yield [input_value]


def representative_dataset_gen_SobelXY():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_Sobel_XY/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=1,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(100):
        yield [input_value]


def representative_dataset_gen_No_Trans():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_No_Trans/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=3,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(100):
        yield [input_value]


def representative_dataset_gen_No_Trans_Gray():
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=['0', '1', '2', '3', '4'],
                              glob_pattern='../Dataset/Videos_No_Trans_Gray/{classname}/*.avi',
                              nb_frames=15,
                              shuffle=True,
                              batch_size=1,
                              target_shape=(224, 224),
                              nb_channel=1,
                              use_frame_cache=True)
    for input_value in tf.data.Dataset.from_tensor_slices(gen.next()[0]).batch(1).take(100):
        yield [input_value]


class TfliteConverterClass:
    """
    Class to convert the h5 models in tflite.
    The input parameters are:

    - modelPath2Convert: the path containing the h5 models to be converted
    - datasetPath: the path of the dataset used for the representative dataset
    - transformation: the transformation applied to the input dataset, it must be compliant with the model path name and the dataset path name
    - experimentalSparsity: optimization option that promotes sparsity in the model, if false default option is used
    - CNNModel: model that has to be trained (default is model 2). Possible choices are in the range 1-3

    """
    def __init__(self,
                 modelPath2Convert: str = None,  # the input path containing the h5 models to be converted
                 datasetPath: str = None,  # the dataset used to create the representative dataset
                 transformation: str = None,  # the transformation applied to the original data
                 experimentalSparsity: bool = False,  # option for the optimization during the conversion promoting the
                 # sparse models. If false keep the default
                 CNNModel: int = 2  # CNN Model
                 ):
        self.inputSize = (15, 224, 224)  # size of the input model.
        channels = 1
        if transformation == "No_Trans":  # No Transformation is the only one with 3 channels (RGB)
            channels = 3
        self.inputSize = self.inputSize + (channels,)  # add the channel to the size of the input videos

        self.es = experimentalSparsity
        # check if the model path to be converted is not none and exists
        assert modelPath2Convert is not None, "Model path to be converted is none"
        assert os.path.exists(modelPath2Convert), "Model Path does not exist"
        self.globModels = glob.glob(modelPath2Convert + '/*')  # take all the files in the path

        self.CNNModel = CNNModel
        if not 1 <= self.CNNModel <= 3:
            print("CNN Model option not valid: default assigned")
            self.CNNModel = 2

        indexToDel = []
        # Remove all the models that are not "h5"
        for i, m in enumerate(self.globModels):
            if "h5" not in m or "Model"+str(self.CNNModel) not in m:
                indexToDel.append(i)
        for ele in sorted(indexToDel, reverse=True):
            del self.globModels[ele]
        assert len(self.globModels) > 0, "Not models to convert"  # check if the list of model is not empty
        self.globModels.sort()

        self.convertedModelPath = "../ModelsTFLite/" + transformation  # assign by default where the converted models will be saved
        if not os.path.exists(self.convertedModelPath):
            os.makedirs(self.convertedModelPath)
        # check if the dataset path exists
        assert os.path.exists(datasetPath), "Dataset Path does not exist"
        self.dataPath = datasetPath

        listOfPossibleTrans = ["Canny", "Sobel_XY", "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]
        if transformation is None:  # if transformation is none assign the default
            print("Transformation assigned by default: No transformation")
            self.transformation = "No_Trans"  # default option No Transformation
        elif transformation not in listOfPossibleTrans:  # if transformation is not in the list of possible transformation assign the default
            print("Requested transformation is not in the list of the possible transformation: No_Tran "
                  "assigned by default")
            self.transformation = "No_Trans"  # default option No Transformation
        else:
            self.transformation = transformation
        # check if transformation is in the model path and in the dataset path
        assert self.transformation in modelPath2Convert, "Input model path must contain the applied transformation"
        assert self.transformation in self.dataPath, "Dataset path must contain the applied transformation"



        self.modelDirTmp = '../ModelsTmp'  # assign by default the folder for the temporary models
        if not os.path.exists(self.modelDirTmp):
            os.mkdir(self.modelDirTmp)

    def create_tflite_fp32(self):  # convert in the tflite fp32 bit model
        for m in self.globModels:
            save_temporary_model(self, m)  # save the h5 model as pb
            name = (m.split('/')[-1]).split('.')[0]  # extract automatically the name of the model
            modelDirTmp = os.path.join(self.modelDirTmp, name)
            converter = tf.lite.TFLiteConverter.from_saved_model(modelDirTmp)  # initialize the converter object
            if self.es:  # is sparsity option is selected then optimize promoting sparsity
                converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
                nameFull = os.path.join(self.convertedModelPath, name + "_ES_fp32.tflite")
            else:  # else default option is selected
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                nameFull = os.path.join(self.convertedModelPath, name + "_fp32.tflite")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float32]  # specify the representation of the operations inside the tflite model
            # invoke the conversion and save the tflite model
            tflite_modelFP32 = converter.convert()
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelFP32)
            print("TFLITE FP32 MODEL {} CREATED".format(name))

    def create_tflite_fp16(self):  # convert in the tflite fp16 bit model
        for m in self.globModels:
            save_temporary_model(self, m)  # save the h5 model as pb
            name = (m.split('/')[-1]).split('.')[0]  # extract automatically the name of the model
            modelDirTmp = os.path.join(self.modelDirTmp, name)
            converter = tf.lite.TFLiteConverter.from_saved_model(modelDirTmp)  # initialize the converter object
            if self.es:  # is sparsity option is selected then optimize promoting sparsity
                converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
                nameFull = os.path.join(self.convertedModelPath, name + "_ES_fp16.tflite")
            else:  # else default option is selected
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                nameFull = os.path.join(self.convertedModelPath, name + "_fp16.tflite")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float16]  # specify the representation of the operations inside the tflite model
            tflite_modelFP16 = converter.convert()
            # invoke the conversion and save the tflite model
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelFP16)
            print("TFLITE FP16 MODEL {} CREATED".format(name))

    def create_tflite_int8(self):  # convert in the tflite int8 bit model with input represented as fp32
        for m in self.globModels:
            save_temporary_model(self, m)  # save the h5 model as pb
            name = (m.split('/')[-1]).split('.')[0]  # extract automatically the name of the model
            modelDirTmp = os.path.join(self.modelDirTmp, name)
            converter = tf.lite.TFLiteConverter.from_saved_model(modelDirTmp)  # initialize the converter object
            if self.es:  # is sparsity option is selected then optimize promoting sparsity
                converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
                nameFull = os.path.join(self.convertedModelPath, name + "_ES_int8.tflite")
            else:  # else default option is selected
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                nameFull = os.path.join(self.convertedModelPath, name + "_int8.tflite")
            converter.post_training_quantize = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                                   tf.lite.OpsSet.SELECT_TF_OPS] # specify the representation of the operations inside the tflite model
            if self.transformation == "Canny":
                rep_dat = representative_dataset_gen_Canny
            elif self.transformation == "Binary":
                rep_dat = representative_dataset_gen_Binary
            elif self.transformation == "Roberts":
                rep_dat = representative_dataset_gen_Roberts
            elif self.transformation == "Sobel_XY":
                rep_dat = representative_dataset_gen_SobelXY
            elif self.transformation == "No_Trans":
                rep_dat = representative_dataset_gen_No_Trans
            else:
                rep_dat = representative_dataset_gen_No_Trans_Gray

            converter.representative_dataset = rep_dat # create the representative dataset
            converter.inference_input_type = tf.int8 # the input and output representations are int8
            converter.inference_output_type = tf.int8
            # invoke the conversion and save the tflite model
            tflite_modelINT8 = converter.convert()
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelINT8)
            print("TFLITE INT8 MODEL {} CREATED".format(name))
