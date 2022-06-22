import shutil
import cv2.cv2 as cv2
import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
from keras.preprocessing.image import img_to_array


def save_temporary_model(self, m):  # function used to create a pb file of the model to be converted
    model = models.load_model(m)  # load the model
    self.inputSize = model.inputs[0].shape  # assign the input size of the model
    # create the concrete function for the conversion of the h5 model in pb format
    run_model = tf.function(lambda x: model(x))
    shape = np.asarray([model.inputs[0].shape[i] for i in range(1, len(model.inputs[0].shape))])
    shape = np.concatenate(([1], shape))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec(shape, model.inputs[0].dtype))
    modelDirTmp = os.path.join(self.modelDirTmp, (m.split('/')[-1]).split('.')[0])
    if os.path.exists(modelDirTmp):
        shutil.rmtree(modelDirTmp)
        os.mkdir(modelDirTmp)
    else:
        os.mkdir(modelDirTmp)
    # convert the model in pb format
    model.save(modelDirTmp, save_format="tf", signatures=concrete_func)
    print("Tmp folder for Model {} conversion created!".format(m.split('/')[-1]))


def load_dataset(self, video):  # function to create the input tensor from a video sample
    shape = (self.inputSize[1], self.inputSize[2])  # image dimensions
    nbFrame = self.inputSize[0]  # number of frames
    nChannels = self.inputSize[3]  # number of channels of the frames
    cap = cv2.VideoCapture(video)  # open the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get the number of total frames
    orig_total = total_frames
    if total_frames % 2 != 0:
        total_frames += 1
    frame_step = np.floor(total_frames / (nbFrame - 1))
    frame_step = max(1, frame_step)
    frames = []
    frame_i = 0

    while True:  # save in a tensor the frames
        grabbed, frame = cap.read()
        if not grabbed:
            break

        frame_i += 1
        if frame_i == 1 or frame_i % frame_step == 0 or frame_i == orig_total:
            frame = cv2.resize(frame, shape)
            if nChannels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = img_to_array(frame)/255.  # rescale the frames between 0 and 1
            frames.append(frame)
        if len(frames) == nbFrame:
            break
    cap.release()
    return np.asarray(frames)


def representative_dataset(self):  # function used to create the representative dataset
    allVideos = glob.glob(self.dataPath + "/*/*")
    np.random.shuffle(allVideos)
    # take 100 samples, if possible, to create the representative dataset
    numOfData = 100
    if len(allVideos) < 100:
        numOfData = len(allVideos)
    data_vid = allVideos[:numOfData]
    for _, ID in enumerate(data_vid):
        X = load_dataset(self, ID)
        yield [X.astype(np.float32)]


class TfliteConverterClass:
    """
    Class to convert the h5 models in tflite.
    The input parameters are:

    - modelPath2Convert: the path containing the h5 models to be converted
    - datasetPath: the path of the dataset used for the representative dataset
    - transformation: the transformation applied to the input dataset, it must be compliant with the model path name and the dataset path name
    - experimentalSparsity: optimization option that promotes sparsity in the model, if false default option is used

    """
    def __init__(self,
                 modelPath2Convert: str = None,  # the input path containing the h5 models to be converted
                 datasetPath: str = None,  # the dataset used to create the representative dataset
                 transformation: str = None,  # the transformation applied to the original data
                 experimentalSparsity: bool = False,  # option for the optimization during the conversion promoting the
                 # sparse models. If false keep the default
                 ):
        self.inputSize = None  # size of the input model. it will be automatically assigned
        self.es = experimentalSparsity
        # check if the model path to be converted is not none and exists
        assert modelPath2Convert is not None, "Model path to be converted is none"
        assert os.path.exists(modelPath2Convert), "Model Path does not exist"
        self.globModels = glob.glob(modelPath2Convert + '/*')  # take all the files in the path
        indexToDel = []
        # Remove all the models that are not "h5"
        for i, m in enumerate(self.globModels):
            if "h5" not in m:
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
            converter.representative_dataset = representative_dataset(self)  # create the representative dataset
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
            converter.representative_dataset = representative_dataset(self)  # create the representative dataset
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
            converter.representative_dataset = representative_dataset(self)  # create the representative dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # specify the representation of the operations inside the tflite model
            converter.inference_input_type = tf.float32  # the input and output representations are fp32
            converter.inference_output_type = tf.float32
            # invoke the conversion and save the tflite model
            tflite_modelINT8 = converter.convert()
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelINT8)
            print("TFLITE INT8 MODEL {} CREATED".format(name))
