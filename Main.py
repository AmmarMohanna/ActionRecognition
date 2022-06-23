from DataProcessingClass import ImagesTransformationClass
from TrainVideoClassifierClass import TrainVideoClassifierClass
from TfliteConverterClass import TfliteConverterClass
from AnalyzeResultsClass import AnalyzeResultsClass

inputImagesPath = "../Dataset/Images"
tran = "No_Trans"
# ImagesTransformationClass(imagesInputPath=inputImagesPath, transformation=tran)

inputVideoPath = "../Dataset/Videos_" + tran
nFolds = 5
for i in range(1, 4):
    TVC = TrainVideoClassifierClass(videoDatasetPath=inputVideoPath, transformation=tran, nFolds=nFolds, CNNModel=i)
    TVC.training()

# modelPath = "../Results/" + tran
# for i in range(1, 4):
#     TFLiteConv = TfliteConverterClass(transformation=tran, modelPath2Convert=modelPath, datasetPath=inputVideoPath, CNNModel=i)
#     TFLiteConv.create_tflite_int8()
#     # TFLiteConv.create_tflite_fp32()
#     # TFLiteConv.create_tflite_fp16()

# for i in range(1, 4):
#     if nFolds > 1:
#         stratifiedKFolds = True
#         xTestpath = "../Results/" + tran + "/X_Test_" + tran + "_StratifiedKFolds_Model{}.pkl".format(i)
#     else:
#         stratifiedKFolds = False
#         xTestpath = "../Results/" + tran + "/X_Test_" + tran + "_NoKFold_Model{}.pkl".format(i)
#     ARC = AnalyzeResultsClass(modelPath=modelPath, videoDatasetPath=inputVideoPath, CNNModel=i,
#                               stratifiedKFolds=stratifiedKFolds, transformation=tran, xTestPath=xTestpath)
#     ARC.testModels()
    # ARC.confusion_matrix()
    # ARC.computeROCandAUC([0, 1])
    # Met = ARC.computeMetrics([0, 1])


# if nFolds > 1:
#     stratifiedKFolds = True
#     xTestpath = "../Results/" + tran + "/X_Test_" + transformation + "_StratifiedKFolds_Model2.pkl"
# else:
#     stratifiedKFolds = False
#     xTestpath = "../Results/" + transformation + "/X_Test_" + transformation + "_NoKFold_Model2.pkl"
# ARC = AnalyzeResultsClass(modelPath=modelPath, videoDatasetPath=inputVideoPath,
#                           stratifiedKFolds=stratifiedKFolds, transformation=transformation, xTestPath=xTestpath)
# ARC.testModels()
# ARC.confusion_matrix()
# ARC.computeROCandAUC([0, 1])
# Met = ARC.computeMetrics([0, 1])
