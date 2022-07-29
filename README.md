# EdgeActionRecognition
This work presented in the folders was used during a research process.
It resulted the publication titled: 

> On Edge Human Action Recognition Using Radar-Based Sensing and Deep Learning

![NN](https://user-images.githubusercontent.com/32446816/181509833-d30ea2ea-fbd6-4a38-b20f-cc7be8428f52.png)

You can find all data here that were collected in the reseach process here: https://www.kaggle.com/datasets/ammarmohanna/edgeactionrecognition

### Here is a quick overview of the files included in this repository

1. Download the data from the aforementioned kaggle repository
![kaggle_repo](https://user-images.githubusercontent.com/32446816/181509190-3cc9ee4f-1f6c-4946-b5a3-14490467251c.png)
<br/><br/>

2. The script *DataProcessingClass* is responsible for generating the 5 different image transformations mentioned in Section 2.B of the paper
<br/><br/>
![Untitled](https://user-images.githubusercontent.com/32446816/181508622-bb9d617a-c0fb-455a-8b84-7bb6b8bd0685.png)
<br/><br/>

3. The script *TrainVideoClassifierClass* is in charge of loading data and training the chosen model
<br/><br/>

4. After training, the script *AnalyzeResultsClass* is responsible for generating the assessments metrics computed in the paper (e.g., confusion matrix, ROC and AUC, metrics, etc.)
![confroc](https://user-images.githubusercontent.com/32446816/181509578-7c085fcc-e9c4-46ae-9eca-55aaf94498cb.png)
<br/><br/>

5. Finally, to deploy on an edge device, you can use the *TfliteConverterClass* to change the format of your trained model to TFLITE compatible.

<br/><br/>
<br/><br/>

For any questions, feel free to ask your questions or contact us directly:

*mohannaammar@gmail.com*
*https://www.linkedin.com/in/ammar-mohanna/*
