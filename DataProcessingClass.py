import cv2.cv2 as cv2
import os
import glob
import numpy as np
import shutil
from scipy import ndimage


# Canny Transformation
def CannyTrans(imPath):
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image=img_gray, threshold1=30, threshold2=30)
    return canny


# Sobel XY Transformation
def SobelXY(imPath):
    mg = cv2.imread(imPath)

    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    sobelXY = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1,
                        ksize=5)
    return sobelXY


# Robert Transformation
def Roberts(imPath):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])
    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)  # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_gr = img_gray / 255.0
    vertical = ndimage.convolve(img_gr, roberts_cross_v)
    horizontal = ndimage.convolve(img_gr, roberts_cross_h)
    roberts = np.sqrt(np.square(horizontal) + np.square(vertical))
    roberts *= 255
    return roberts


# Binary Transformation
def Binary(imPath):
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    pixel_values = img_gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_gray.shape)
    binary = ndimage.median_filter(segmented_image, size=30)
    minn = np.min(binary)
    binary[binary < minn + 1] = 0
    binary[binary >= minn + 1] = 255
    return binary


# No Transformation, images saved in gray scale
def NoTransGray(imPath):
    mg = cv2.imread(imPath)
    gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    return gray


# No Transformation
def NoTrans(imPath):
    noTrans = cv2.imread(imPath)
    return noTrans


class ImagesTransformationClass:
    """
    This class leads to transform the input images based on well-defined technique and save the results in a video:

    - imagesInputPath: the input path from where the class will take the images to be transformed
    - transformation: the transformation to be applied to the images. The possible choices are: Canny, Sobel XY, Roberts, Binary, No Transformation in gray scale, No Transformation
    """
    def __init__(self,
                 imagesInputPath: str,  # the input path of the images, the images must be divided in classes
                 transformation: str = None  # the transformation to be applied to the images
                 ):
        listOfPossibleTrans = ["Canny", "Sobel_XY", "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]
        if transformation is None:
            print("Transformation assigned by default: No transformation")
            self.transformation = "No_Trans"  # default option No Transformation
        elif transformation not in listOfPossibleTrans:
            print("Requested transformation is not in the list of the possible transformation: No transformation " 
                  "assigned by default")
            self.transformation = "No_Trans"
        else:
            print("Transforming data by {} transformation".format(transformation))
            self.transformation = transformation

        assert os.path.exists(imagesInputPath), "Input images path does not exist"
        self.globInputImagesPath = glob.glob(imagesInputPath + "/*/*")  # it will contain all the sub-folders for each
        # class
        self.globInputImagesPath.sort()

        self.classes = list(np.unique([i.split('/')[-2] for i in self.globInputImagesPath]))  # find the name of the
        # classes

        self.videosTransOutputPath = "../Dataset/Videos_" + self.transformation  # default output video folder
        if not os.path.exists(self.videosTransOutputPath):  # create the output folder if it does not exist
            print("Output folder does not exist: creating a new one")
            os.makedirs(self.videosTransOutputPath)
            for c in range(len(self.classes)):
                os.mkdir(os.path.join(self.videosTransOutputPath, str(c)))
        else:  # delete all the contents of the old folder and create a new one
            print("Output folder already exist: data will be overwritten")
            shutil.rmtree(os.path.join(self.videosTransOutputPath))
            os.makedirs(self.videosTransOutputPath)
            for c in range(len(self.classes)):
                os.mkdir(os.path.join(self.videosTransOutputPath, str(c)))

        self.TransformImages()  # invoke the function for transforming the images

    def TransformImages(self):
        for fol in self.globInputImagesPath:  # for each sub-folder
            indexClass = self.classes.index(fol.split('/')[-2])  # find the index of the corresponding class in the
            # classes list
            objectNum = int(fol.split('/')[-1])  # find the number of the sub-folder
            imgList = os.listdir(fol)  # list of the images in each sub-folder
            imgList.sort()
            imgForVideo = []  # this list collects the transformed images to be transformed in a video
            for im in imgList:
                imPathTmp = os.path.join(fol, im)
                #  Call the method to transform the images
                if self.transformation == "Canny":
                    imgTrans = CannyTrans(imPathTmp)
                elif self.transformation == "Sobel_XY":
                    imgTrans = SobelXY(imPathTmp)
                elif self.transformation == "Binary":
                    imgTrans = Binary(imPathTmp)
                elif self.transformation == "Roberts":
                    imgTrans = Roberts(imPathTmp)
                elif self.transformation == "No_Trans_Gray":
                    imgTrans = NoTransGray(imPathTmp)
                else:
                    imgTrans = NoTrans(imPathTmp)
                imgForVideo.append(imgTrans)

            self.CreateVideo(imgForVideo, indexClass, objectNum)  # invoke this function to save the transformed video

    def CreateVideo(self, frames, classFolder, objectNum):  # function that creates the video, the inputs are the list
        # of the images that form the video, the corresponding class, the number of the input images sub-folders
        fps = 30
        height, width, *channels = frames[0].shape
        if objectNum < 10:
            numStr = "000" + str(objectNum)
        elif 10 <= objectNum < 100:
            numStr = "00" + str(objectNum)
        elif 100 <= objectNum < 1000:
            numStr = "0" + str(objectNum)
        else:
            numStr = str(objectNum)
        name = os.path.join(self.videosTransOutputPath, str(classFolder), str(self.classes[classFolder]) + numStr + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if len(channels) == 0:
            out = cv2.VideoWriter(name, fourcc, fps, (width, height), 0)
        else:
            out = cv2.VideoWriter(name, fourcc, fps, (width, height))
        for i in range(len(frames)):
            tmp = np.uint8(frames[i])
            out.write(tmp)
        out.release()
