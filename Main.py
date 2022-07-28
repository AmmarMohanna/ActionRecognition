from DataProcessingClass import ImagesTransformationClass
from TrainVideoClassifierClass import TrainVideoClassifierClass
from TfliteConverterClass import TfliteConverterClass
from AnalyzeResultsClass import AnalyzeResultsClass

inputImagesPath = "../Dataset/Images"
tran = "No_Trans"

inputVideoPath = "../Dataset/Videos_" + tran
nFolds = 5
for i in range(1, 4):
    TVC = TrainVideoClassifierClass(videoDatasetPath=inputVideoPath, transformation=tran, nFolds=nFolds, CNNModel=i)
    TVC.training()

