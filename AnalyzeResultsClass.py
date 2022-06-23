import glob
import os
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_video import VideoFrameGenerator
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


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


def visualizeConfMatrix(self, cm):  # Function plotting the confusion matrix
    cm_df = pd.DataFrame(cm, index=self.classesNames, columns=self.classesNames)
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

    sns.set(font_scale=2.5)
    s = np.sum(cm, axis=1)
    sns.heatmap(cm_df / s, annot=cm, cmap="Greys", annot_kws={"size": 36}, cbar=False, linewidths=1,
                linecolor='black',
                fmt='g', square=True)
    plt.ylabel('True labels', fontsize=28, fontweight="bold")
    plt.xlabel('Predicted label', fontsize=28, fontweight="bold")

    if "Model" in self.globModels[0]:
        nameModel = " Model" + self.globModels[0][self.globModels[0].find("Model") + 5]
    else:
        nameModel = ""
    if self.stratifiedKFolds:
        title = "Confusion Matrix KFolds " + self.transformation + nameModel
    else:
        title = "Conf Matrix " + self.transformation + nameModel
    plt.title(title, fontsize=32, fontweight="bold")
    plt.show()


def compute_ROC(self, y_score, true_labels, labelsOfPositive):
    cl = np.asarray([int(i) for i in self.classes])
    y_test = label_binarize(true_labels, classes=cl)
    y_test_tmp = []
    y_score_tmp = []
    for i in labelsOfPositive:
        y_test_tmp = np.concatenate((y_test_tmp, y_test[:, i]))
        y_score_tmp = np.concatenate((y_score_tmp, y_score[:, i]))
    fpr, tpr, _ = roc_curve(y_test_tmp, y_score_tmp, drop_intermediate=False)
    return fpr, tpr


def visualizeROCandAUC(self, fpr, tpr, roc_auc):  # Function plotting the ROC and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(fpr))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(fpr)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(fpr)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], label="Average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]),
             color="black", linestyle=":", linewidth=5)

    colors = ["aqua", "darkorange", "cornflowerblue", "deeppink", "limegreen", "darksalmon", "orchid", "gold", "grey",
              "navy"]
    for i in range(len(fpr)-1):
        c = i % len(colors)
        if self.stratifiedKFolds:
            plt.plot(fpr[i], tpr[i], color=colors[c], linewidth=5,
                     label="ROC curve of Fold {0} (area = {1:0.3f})".format(i, roc_auc[i]))
        else:
            plt.plot(fpr[i], tpr[i], color=colors[c], linewidth=5,
                     label="ROC curve (area = {0:0.3f})".format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=36)
    plt.ylabel("True Positive Rate", fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if "Model" in self.globModels[0]:
        nameModel = " Model" + self.globModels[0][self.globModels[0].find("Model") + 5]
    else:
        nameModel = ""
    if self.stratifiedKFolds:
        title = "ROC and AUC KFolds " + self.transformation + nameModel
    else:
        title = "ROC and AUC " + self.transformation + nameModel
    plt.title(title, fontsize=40)
    plt.legend(loc="lower right", fontsize=24)
    plt.show()


def generator(self, inputSize, data_to_take):
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=self.classes,
                              glob_pattern=self.dataPath,
                              nb_frames=inputSize[0],
                              shuffle=False,
                              batch_size=1,
                              target_shape=inputSize[1:3],
                              nb_channel=inputSize[3],
                              use_frame_cache=False,
                              _test_data=data_to_take)
    return gen


class AnalyzeResultsClass:
    """
    This class is used to show the results concerning the accuracy of the model on the test data, and to plot the confusion matrices and ROC and AUC curves.
    The input parameters are:

    - modelPath: the path where model to be tested are saved
    - videoDatasetPath: the dataset path that will be used for plotting the results
    - stratifiedKFolds: if true the models trained with KFolds strategy will be analyzed, else the models trained without the KFolds will be taken into consideration
    - xTestPath: the path/file containing the names of the data that form the test set. It can be a pickle file or a path of the folder with the data to be tested
    - transformation: the transformation applied to the original dataset. It is used to check if the model path, the video path, and the test path are consistent
    - classesNames: list containing the names of the classes for plotting the results
    - CNNModel: model that has to be trained (default is model 2). Possible choices are in the range 1-3
    """
    def __init__(self,
                 modelPath: str = None,  # The model to be analyzed
                 videoDatasetPath: str = None,  # The path with the whole dataset
                 stratifiedKFolds: bool = True,  # variable that identifies if stratified KFolds has been used
                 xTestPath: str = None,  # path containing the names of data used as test set
                 transformation: str = None,  # transformation applied to the original dataset
                 classesNames: list = None,  # name of classes, used for plotting the results
                 CNNModel: int = 2
                 ):
        assert os.path.exists(videoDatasetPath), "Dataset does not exist"  # check if the dataset exist
        self.dataPath = videoDatasetPath + '/{classname}/*.avi'  # used by keras video generator
        self.globInputVideos = glob.glob(videoDatasetPath + "/*/*")  # get all the names of data
        self.globInputVideos.sort()
        self.labels = np.asarray([int(i.split('/')[-2]) for i in self.globInputVideos])  # get the labels of data automatically
        self.classes = list(np.unique([i.split('/')[-2] for i in self.globInputVideos]))  # get the classes
        self.classes.sort()
        self.nClasses = len(self.classes)  # get the number of classes
        # check if the path of the test set exists
        assert os.path.exists(xTestPath), " Test path does not exist: provide the file with the data to be tested: " \
                                          "can be the folder path containing the data or a pickle file containing" \
                                          " the names of the data path to be tested"
        if xTestPath.endswith("pkl"):  # load the pickle file if exists
            with open(xTestPath, 'rb') as f:
                xTestDict = pkl.load(f)
                self.X_test = xTestDict["X_Test"]
                self.y_test = xTestDict["y_Test"]
                f.close()
        else:  # else get all the files that will be used as test set
            self.X_test = glob.glob(xTestPath+"/*/*")
            self.y_test = np.asarray([int(i.split('/')[-2]) for i in self.X_test])

        # check if the models' path exists
        assert os.path.exists(modelPath), "Model path does not exist"
        self.globModels = glob.glob(modelPath + '/*')

        self.CNNModel = CNNModel
        if not 1 <= self.CNNModel <= 3:
            print("CNN Model option not valid: default assigned")
            self.CNNModel = 2

        # delete useless files from globModels
        indexToDel = []
        for i, m in enumerate(self.globModels):
            if ".h5" not in m or "Model"+str(self.CNNModel) not in m:
                indexToDel.append(i)
            else:
                if stratifiedKFolds:  # the names of the models must contain "Fold" string
                    if "Fold" not in m:
                        indexToDel.append(i)
                else:  # else remove the files that contain "Fold" string
                    if "Fold" in m:
                        indexToDel.append(i)
        for ele in sorted(indexToDel, reverse=True):
            del self.globModels[ele]
        assert len(self.globModels) > 0, "There are not models to test"
        self.globModels.sort()

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
        # check if transformation is in the model path, in the dataset path, and in the test data path
        assert self.transformation in modelPath, "Input model path must contain the applied transformation"
        assert self.transformation in self.dataPath, "Dataset path must contain the applied transformation"
        assert self.transformation in xTestPath, "Test data path must contain the applied transformation"

        # assign the classes names for plotting the results
        if classesNames is not None:
            self.classesNames = classesNames
        else:
            print("Classes names assigned by default: Bed, Fall, Sit, Stand, Walk")
            self.classesNames = ["Bed", "Fall", "Sit", "Stand", "Walk"]
        # and check if the list length is equal to the number of classes
        assert len(self.classesNames) == len(self.classes), "Length of classes names different from length of classes"

        if stratifiedKFolds:
            # if KFolds has been used, check if the models have a corresponding dataset to be tested
            assert len(self.X_test) == len(self.globModels), " Number of datasets different from the number of folds"
            print("Test {} models on KFolds".format(self.transformation))
        else:
            # else, check if there is only one list of data to be tested
            assert not any(isinstance(el, list) for el in self.X_test), " The list must have only one dimension"
            print("Test {} model without KFolds".format(self.transformation))
        self.stratifiedKFolds = stratifiedKFolds

        self.inputSize = (15, 224, 224)  # size of the input model.
        channels = 1
        if transformation == "No_Trans":  # No Transformation is the only one with 3 channels (RGB)
            channels = 3
        self.inputSize = self.inputSize + (channels,)  # add the channel to the size of the input videos

    def testModels(self):  # function used to print the accuracy on the test set for each model contained in globModels
        testAccList = []
        for mod in self.globModels:
            model = deep_network(self)
            model.load_weights(mod)
            inputSize = model.inputs[0].shape[1:]  # take the input tensor size
            print("Testing {}".format(mod.split('/')[-1]))
            if self.stratifiedKFolds:
                # take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold")+4])
                xT = self.X_test[indexTmp]
                yT = self.y_test[indexTmp]
            else:
                xT = self.X_test
                yT = self.y_test
            testGen = generator(self, inputSize=inputSize, data_to_take=xT)  # create the generator for testing the model
            pred = model.predict(testGen, steps=len(xT))
            y_pred = np.argmax(pred, axis=1)
            true_labels = yT
            testAccTmp = 1 - np.count_nonzero(y_pred - true_labels) / len(y_pred)
            print('Test accuracy:', testAccTmp)
            testAccList.append(testAccTmp)
        testAccArray = np.asarray(testAccList)
        modelName = self.globModels[0].split('_')[-1].split('.')[0]
        print("Average Accuracy {}: {} +- {}".format(modelName, np.mean(testAccArray), np.std(testAccArray)))
        return testAccList

    def confusion_matrix(self):  # Function used to plot the confusion matrix
        cm = np.zeros((self.nClasses, self.nClasses))
        if self.stratifiedKFolds:
            print("Computing Confusion Matrix Stratified KFolds {}".format(self.transformation))
        else:
            print("Computing Confusion Matrix {}".format(self.transformation))
        for mod in self.globModels:
            model = deep_network(self)
            model.load_weights(mod)
            inputSize = model.inputs[0].shape[1:]  # Take the input tensor size
            if self.stratifiedKFolds:
                # Take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.X_test[indexTmp]
                yT = self.y_test[indexTmp]
            else:
                xT = self.X_test
                yT = self.y_test
            testGen = generator(self, inputSize=inputSize, data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            y_pred = np.argmax(pred, axis=1)
            true_labels = yT
            cm += confusion_matrix(true_labels, y_pred)
            if not self.stratifiedKFolds:  # If the KFolds technique is not used plot the confusion matrix for each model,
                # else sum the results for each fold
                visualizeConfMatrix(self, cm)
                cm = np.zeros((self.nClasses, self.nClasses))  # Reinitialize the confusion matrix
        if self.stratifiedKFolds:  # If the KFolds technique is used plot the cumulated confusion matrices
            visualizeConfMatrix(self, cm)

    def computeROCandAUC(self, labelsOfPositive):  # function to plot the ROC and AUC curves. It takes in input the list
        # of the classes that must be considered as positive samples.
        # Initialization of dictionaries used to print the ROC and AUC curves
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Check if the list of positive is not empty
        assert len(labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.nClasses, "the labels of positive samples are bigger than the number of classes"
        if self.stratifiedKFolds:
            print("Computing ROC and AUC Curves Stratified KFolds {}".format(self.transformation))
        else:
            print("Computing ROC and AUC Curves {}".format(self.transformation))
        ii = 0
        for mod in self.globModels:
            model = deep_network(self)
            model.load_weights(mod)
            inputSize = model.inputs[0].shape[1:]  # Take the input tensor size
            if self.stratifiedKFolds:
                # Take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.X_test[indexTmp]
                yT = self.y_test[indexTmp]
            else:
                xT = self.X_test
                yT = self.y_test
            testGen = generator(self, inputSize=inputSize, data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            true_labels = yT
            fpr_tmp, tpr_tmp = compute_ROC(self, pred, true_labels, labelsOfPositive)  # Compute the false positive and true positive rates
            fpr[ii] = fpr_tmp
            tpr[ii] = tpr_tmp
            roc_auc[ii] = auc(fpr_tmp, tpr_tmp)  # Compute the AUC curve
            if not self.stratifiedKFolds:  # If the KFolds technique is not used plot the ROC and AUC for each model, else plot the results for all the folds
                visualizeROCandAUC(self, fpr, tpr, roc_auc)
            else:
                ii += 1
        if self.stratifiedKFolds:
            visualizeROCandAUC(self, fpr, tpr, roc_auc)

    def computeMetrics(self, labelsOfPositive):  # Function to compute al the metrics: precision, recall, specifity,
        # false positive rate, false negative rate, accuracy, and f1 score. It takes in input the list of the classes
        # that must be considered as positive samples.
        # Initialization of the results lists
        precision = []
        recall = []
        specifity = []
        FPR = []
        FNR = []
        accuracy = []
        f1score = []
        # Check if the list of positive is not empty
        assert len(labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.nClasses, "the labels of positive samples are bigger than the number of classes"
        for i, mod in enumerate(self.globModels):
            model = deep_network(self)
            model.load_weights(mod)
            inputSize = model.inputs[0].shape
            if self.stratifiedKFolds:
                # Take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.X_test[indexTmp]
                yT = self.y_test[indexTmp]
            else:
                xT = self.X_test
                yT = self.y_test
            testGen = generator(self, inputSize=inputSize, data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            pred_labels = np.argmax(pred, axis=1)
            true_labels = yT
            predBin = []
            # Append 1 if the predictions ar contained in the positive labels list, else 0
            for jj in pred_labels:
                if jj in labelsOfPositive:
                    predBin.append(1)
                else:
                    predBin.append(0)
            predBin = np.asarray(predBin)
            true_labelsBin = []
            # Append 1 if the true labels ar contained in the positive labels list, else 0
            for jj in true_labels:
                if jj in labelsOfPositive:
                    true_labelsBin.append(1)
                else:
                    true_labelsBin.append(0)
            true_labelsBin = np.asarray(true_labelsBin)
            tn, fp, fn, tp = confusion_matrix(true_labelsBin, predBin).ravel()  # compute the true positive, the false positive,
            # the true negative, and false negative rates.
            # Append the results to the corresponding lists
            precision.append(tp/(tp+fp))
            recall.append(tp/(tp+fn))
            specifity.append(tn/(tn+fp))
            FPR.append(fp/(fp+tn))
            FNR.append(fn/(fn+tp))
            accuracy.append((tp+tn)/(tp+tn+fn+fp))
            f1score.append(2*tp/(2*tp+fp+fn))
        # Create a dictionary for the results
        Metrics = {"Prec": np.mean(precision), "Rec": np.mean(recall), "Spec": np.mean(specifity), "FPR": np.mean(FPR),
                   "FNR": np.mean(FNR), "Acc": np.mean(accuracy), "F1": np.mean(f1score)}
        return Metrics
