from DataProcessingClass import ImagesTransformationClass
from TrainVideoClassifierClass import TrainVideoClassifierClass
from TfliteConverterClass import TfliteConverterClass
from AnalyzeResultsClass import AnalyzeResultsClass

inputImagesPath = "../Dataset/Images"
transformation = "Canny"
# ImagesTransformationClass(imagesInputPath=inputImagesPath, transformation=transformation)

inputVideoPath = "../Dataset/Videos_" + transformation
nFolds = 1
# TVC = TrainVideoClassifierClass(videoDatasetPath=inputVideoPath, transformation=transformation, nFolds=nFolds)
# TVC.training()

modelPath = "../Results/" + transformation
# TFLiteConv = TfliteConverterClass(transformation=transformation, modelPath2Convert=modelPath, datasetPath=inputVideoPath)
# TFLiteConv.create_tflite_fp32()


if nFolds > 1:
    stratifiedKFolds = True
    xTestpath = "../Results/" + transformation + "/X_Test_" + transformation + "_StratifiedKFolds_Model2.pkl"
else:
    stratifiedKFolds = False
    xTestpath = "../Results/" + transformation + "/X_Test_" + transformation + "_NoKFold_Model2.pkl"
ARC = AnalyzeResultsClass(modelPath=modelPath, videoDatasetPath=inputVideoPath,
                          stratifiedKFolds=stratifiedKFolds, transformation=transformation, xTestPath=xTestpath)
ARC.testModels()
ARC.confusion_matrix()
# ARC.computeROCandAUC([0, 1])
# Met = ARC.computeMetrics([0, 1])
