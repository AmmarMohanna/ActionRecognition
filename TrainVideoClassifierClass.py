import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from keras_video import VideoFrameGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from CNNs_architecture import *

def oneFoldTraining(self):  # function to train the model with only one split of the data
    X = self.globInputVideos
    y = self.labels
    # 60% of data used for training, 20% for validation and 20% for test
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y, train_size=0.6,
                                                      random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, stratify=y_val, train_size=0.5, random_state=123)
    # Use the keras video generator for the training of the model
    train = generator(self, data_to_take=X_train, bs=self.bs)
    val = generator(self, data_to_take=X_val, bs=self.bs)
    test = generator(self, data_to_take=X_test, bs=1, sh=False)
    # name of the file to be saved containing the test data for further analysis
    name2save = os.path.join(self.saveResPath, "X_Test_{}_NoKFold_Model{}.pkl".format(self.transformation,
                                                                                      self.CNNModel))
    # Save the test dataset
    testDict = {"X_Test": X_test, "y_Test": y_test}
    with open(name2save, "wb") as f:
        pickle.dump(testDict, f)
        f.close()
    model = deep_network(self)  # create the model
    optimizer = tf.keras.optimizers.Adam(self.lr)  # definition of the optimizer, Adam is chosen
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, steps_per_epoch=train.files_count // self.bs, validation_data=val,
                        validation_steps=val.files_count // self.bs, verbose=1, epochs=self.ep,
                        callbacks=self.callbacks)  # fitting the model using the generators and the callback function
    fine_model = tf.keras.models.clone_model(model)  # cloning the model for fine-tuning procedure
    lrEnd = model.optimizer.lr.numpy()  # get the learning rate on the last trained epoch
    optimizerFine = tf.keras.optimizers.Adam(lrEnd * 0.07)  # define the new optimizer for the fine-tuning
    fine_model.compile(optimizerFine, 'categorical_crossentropy', metrics=['accuracy'])
    fine_model.set_weights(model.get_weights())  # copy the weights of the trained model
    start_epochs = len(history.history['loss'])
    numOfEpochsFineTuning = start_epochs + 30  # fine-tune for 30 epochs
    _ = fine_model.fit(train, steps_per_epoch=train.files_count // self.bs, verbose=1, epochs=numOfEpochsFineTuning,
                       callbacks=self.callbacksFineTuning, initial_epoch=start_epochs)  # fitting the fine-tuned model
    loss_fine, acc_fine = fine_model.evaluate(test)  # print the accuracy on the test set
    print('Test accuracy: ', acc_fine)
    # save the fine-tuned model
    model2save = os.path.join(self.saveResPath, "{}_Model{}.h5".format(self.transformation, self.CNNModel))
    fine_model.save_weights(model2save)


def stratifiedKFoldTraining(self):  # function to train the models along the folds
    X = self.globInputVideos
    y = self.labels
    skf = StratifiedKFold(n_splits=self.nFolds, shuffle=True, random_state=123)  # create the stratified KFolds object
    fo = 0
    testDict = {"X_Test": [], "y_Test": []}  # dictionary to save all the test sets to be used for further analyses
    for train_index, test_index in skf.split(X, y):  # for loop along the folds
        print("TRAINING FOLD: ", fo)
        X_train = [X[i] for i in train_index]  # take the training data for the fold
        y_train = y[train_index]  # and the label of the training data
        X_tr, X_val = train_test_split(X_train, shuffle=True, stratify=y_train, train_size=60 / 80,
                                       random_state=123 + fo * 834)  # split the training data in training and validation
        X_test = [X[i] for i in test_index]  # take the test data
        y_test = np.asarray([y[i] for i in test_index])  # and the labels of the test set
        testDict["X_Test"].append(X_test)
        testDict["y_Test"].append(y_test)
        # Use the keras video generator for the training of the models
        train = generator(self, data_to_take=X_tr, bs=self.bs)
        valid = generator(self, data_to_take=X_val, bs=self.bs)
        test = generator(self, data_to_take=X_test, bs=1, sh=False)
        model = deep_network(self)  # create the model
        optimizer = tf.keras.optimizers.Adam(self.lr)  # definition of the optimizer, Adam is chosen
        model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train, steps_per_epoch=train.files_count // self.bs, validation_data=valid,
                            validation_steps=valid.files_count // self.bs, verbose=1, epochs=self.ep,
                            callbacks=self.callbacks)  # fitting the model using the generators and the callback function
        fine_model = tf.keras.models.clone_model(model)  # cloning the model for fine-tuning procedure
        lrEnd = model.optimizer.lr.numpy()  # get the learning rate on the last trained epoch
        optimizerFine = tf.keras.optimizers.Adam(lrEnd * 0.07)  # define the new optimizer for the fine-tuning
        fine_model.compile(optimizerFine, 'categorical_crossentropy', metrics=['accuracy'])
        fine_model.set_weights(model.get_weights())  # copy the weights of the trained model
        start_epochs = len(history.history['loss'])
        numOfEpochsFineTuning = start_epochs + 30  # fine-tune for 30 epochs
        _ = fine_model.fit(train, steps_per_epoch=train.files_count // self.bs, verbose=1,
                           epochs=numOfEpochsFineTuning, callbacks=self.callbacksFineTuning,
                           initial_epoch=start_epochs)  # fitting the fine-tuned model
        loss_fine, acc_fine = fine_model.evaluate(test)  # print the accuracy on the test set
        print('Test accuracy: ', acc_fine)
        # save the fine-tuned model for each fold
        model2save = os.path.join(self.saveResPath, "{}_Fold{}_Model{}.h5".format(
            self.transformation, fo, self.CNNModel))
        fine_model.save(model2save)
        fo += 1
        if fo == self.nFolds:
            break
    # save the dictionary of the test data
    name2save = os.path.join(self.saveResPath, "X_Test_{}_StratifiedKFolds_Model{}.pkl".format(self.transformation,
                                                                                               self.CNNModel))
    with open(name2save, "wb") as f:
        pickle.dump(testDict, f)
        f.close()


def generator(self, data_to_take, bs=10, sh=True):  # function that returns the keras video generator
    gen = VideoFrameGenerator(rescale=1 / 255.,
                              classes=self.classes,
                              glob_pattern=self.dataPath,
                              nb_frames=self.size[0],
                              shuffle=sh,
                              batch_size=bs,
                              target_shape=self.size[1:3],
                              nb_channel=self.channels,
                              use_frame_cache=True,
                              _test_data=data_to_take)
    return gen


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    # class for reducing the learning rate every 5 epochs, it's called automatically by the fit function
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr * 0.9
            self.model.optimizer.lr.assign(new_lr)


class LearningRateReducerCbFineTuning(tf.keras.callbacks.Callback):
    # class for reducing the learning rate in the fine-tuning procedure every 7 epochs, it's called automatically by the fit function
    def on_epoch_end(self, epoch,  logs={}):
        if epoch % 7 == 0:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr * 0.5
            self.model.optimizer.lr.assign(new_lr)


class TrainVideoClassifierClass:
    """
    This class is used to train the three models.
    The input parameters are:

    - params: dictionary that contains the parameter for the training, i.e. batch size, number of epochs, learning rate, input size, patience for the early stopping. If some of the parameters is not given, default values have been assigned
    - videoDatasetPath: the path to the videos used for the training procedure. The subfolders must be the classes of the dataset containing the videos.
    - nFolds: represents the number of folds used for the KFolds technique. If =1 then the dataset is just split in training, validation, and test
    - transformation: which transformation is used on the input data. Must be compliant with the name of videoDatasetPath
    - CNNModel: model that has to be trained (default is model 2). Possible choices are in the range 1-3

    The training consists of a training with the given learning rate and then 30 epochs of fine-tuning with a reduced learning rate
    """

    def __init__(self,
                 params: dict = None,  # contains the parameters for the training procedure
                 videoDatasetPath: str = None,  # input dataset path
                 nFolds: int = 1,  # number of folds, if >1 stratified KFolds technique is adopted
                 transformation: str = None,  # transformation that has been applied to the input data, default No_Transformation. It must be compliant with the name of the videoDatasetPath
                 CNNModel: int = 2
                 ):
        if params is not None:
            if "batch_size" not in params.keys():
                print("Batch size assigned by default: 10")
                self.bs = 10  # default batch size is 10
            else:
                self.bs = params["batch_size"]
            if "epochs" not in params.keys():
                print("Number of epochs assigned by default: 200")
                self.ep = 200  # default number of epochs is 200
            else:
                self.ep = params["epochs"]
            if "learning_rate" not in params.keys():
                print("Learning Rate assigned by default: 7e-5")
                self.lr = 7e-5  # default learning rate is 7e-5
            else:
                self.lr = params["learning_rate"]
            if "size" not in params.keys():
                print("Input Size missing: default assigned")
                self.size = (15, 224, 224)  # default input size of the videos, the first dimension represents the frames,
                # the second and the third the dimension of the images. The number of channels will be automatically assigned later
            else:
                if not len(params["size"]) == 3:
                    print("Wrong input size: default assigned")
                    self.size = (15, 224, 224)
                else:
                    self.size = params["size"]
            if "patience" not in params.keys():
                print("Patience for early stop assigned by default: 10")
                self.patience = 10  # default patience for the early stop criterion
            else:
                self.patience = params["patience"]
        else:
            # default parameters assignment
            print("Batch size assigned by default: 10")
            self.bs = 10
            print("Number of epochs assigned by default: 200")
            self.ep = 200
            print("Learning Rate assigned by default: 7e-5")
            self.lr = 7e-5
            print("Input Size missing: default assigned")
            self.size = (15, 224, 224)
            print("Patience for early stop assigned by default: 10")
            self.patience = 10
        assert os.path.exists(videoDatasetPath), "Dataset does not exist"  # check if video path exists

        self.dataPath = videoDatasetPath + '/{classname}/*.avi'  # path used by the video generator
        self.globInputVideos = glob.glob(videoDatasetPath + "/*/*")  # globInputVideos contains all the data as paths
        self.globInputVideos.sort()
        self.labels = np.asarray([int(i.split('/')[-2]) for i in self.globInputVideos])  # automatic labels extraction
        self.classes = list(np.unique([i.split('/')[-2] for i in self.globInputVideos]))  # classes of the dataset
        self.classes.sort()
        self.nClasses = len(self.classes)  # number of classes

        listOfPossibleTrans = ["Canny", "Sobel_XY", "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]  # list of possible transformations
        if transformation is None:  # if transformation is not given then assigned the default
            print("Transformation assigned by default: No transformation")
            self.transformation = "No_Trans"  # default option No Transformation
            assert self.transformation in videoDatasetPath, "Input Video path must contain the applied transformation"
        elif transformation not in listOfPossibleTrans:  # if the given transformation does not exist then assigned the default
            print("Requested transformation is not in the list of the possible transformation: No_Tran "
                  "assigned by default")
            self.transformation = "No_Trans"
            # check if the transformation is the same of input videos
            assert self.transformation in videoDatasetPath, "Input Video path must contain the applied transformation"
        else:
            self.transformation = transformation
            assert self.transformation in videoDatasetPath, "Input Video path must contain the applied transformation"
        self.channels = 1
        if transformation == "No_Trans":  # No Transformation is the only one with 3 channels (RGB)
            self.channels = 3
        self.size = self.size + (self.channels,)  # add the channel to the size of the input videos

        tmp1 = "../Results"  # Assign by default the path where the models will be saved
        if not os.path.exists(tmp1):
            os.mkdir(tmp1)
        tmp2 = os.path.join(tmp1, self.transformation)
        if not os.path.exists(tmp2):
            os.mkdir(tmp2)
        self.saveResPath = tmp2

        if nFolds > 1:  # check if folds is greater than 1 for the stratified KFolds technique
            print("Stratified K-Fold technique applied during training")
            self.nFolds = nFolds  # if nFolds > 1 then stratified K-Folds technique will be applied
        else:
            print("Dataset will be split in training 60%, validation 60%, and test 20%")
            self.nFolds = 1  # if not, the data will be automatically split in training, validation, and test

        self.tensorboardLogDir = "../logdir"
        if not os.path.exists(self.tensorboardLogDir):
            os.mkdir(self.tensorboardLogDir)
        # callbacks for the fit function. The first one is use during the first training, applying a early stop criterion,
        # and saving the log for visualizing the tensorboard
        self.callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, monitor='val_loss'), LearningRateReducerCb(),
                          tf.keras.callbacks.TensorBoard(log_dir=self.tensorboardLogDir)]
        # The second callback is used during the fine-tuning procedure to reduce the learning rate along the epochs
        self.callbacksFineTuning = [LearningRateReducerCbFineTuning(), tf.keras.callbacks.TensorBoard(log_dir=self.tensorboardLogDir)]

        self.LSTMNeurons = 128
        self.DenseNeurons = 64

        self.CNNModel = CNNModel
        if not 1 <= self.CNNModel <= 3:
            print("CNN Model option not valid: default assigned")
            self.CNNModel = 2

    def training(self):  # training function
        if self.nFolds == 1:
            oneFoldTraining(self)
        else:
            stratifiedKFoldTraining(self)
