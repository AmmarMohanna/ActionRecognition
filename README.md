# EdgeActionRecognition
This work presented in the folders was used during a research process.
It resulted the publication titled: 

> Edge Multi-Class Human Action Recognition System Based on Deep Learning Using a Low-Cost Radar for Indoor Safety

You can find all data here that were collected in the reseach process here: https://www.kaggle.com/datasets/ammarmohanna/edgeactionrecognition

### Here is a quick overview of the files included in this repository

1. Download the data from the aforementioned kaggle repository
2. The script *DataProcessingClass* is responsible for generating the 5 different image transformations mentioned in Section 2.B of the paper
3. The script *TrainVideoClassifierClass* is in charge of loading data and training the chosen model
4. After training, the script *AnalyzeResultsClass* is responsible for generating the assessments metrics computed in the paper (e.g., confusion matrix, ROC and AUC, metrics, etc.)
5. Finally, to deploy on an edge device, you can use the *TfliteConverterClass* to change the format of your trained model to TFLITE compatible.

For any questions, feel free to ask your questions or contact us directly:



*mohannaammar@gmail.com*
*https://www.linkedin.com/in/ammar-mohanna/*
