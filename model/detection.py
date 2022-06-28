# from InceptionResnetV1 import inception_resnet_v1
import os
import pandas as pd
import os
import cv2 as cv
import subprocess
import shutil

# USAR ESSA VERSÃO DO TENSORFLOW !pip install tensorflow==1.13.1
originalFolder = "original/"
croppedFolder = "cropped/"
originalFolderPath = ""
croppedFolderPath = ""
personFolderPath = ""


def generate_paths(imagesPath="./data/"):
    global originalFolderPath
    global croppedFolderPath
    originalFolderPath = imagesPath + originalFolder
    croppedFolderPath = imagesPath + croppedFolder


def generateCroppedImagesFromVideo(
    videoFilePath, personName="John_Doe", imagesPath="./data/"
):
    personOriginalFolderPath = imagesPath + originalFolder + personName
    personCroppedFolderPath = imagesPath + croppedFolder + personName
    if not os.path.exists(personOriginalFolderPath):
        os.mkdir(f"{personOriginalFolderPath}/")

    capture = cv.VideoCapture(videoFilePath)

    if capture.isOpened():
        success, image = capture.read()
        count = 0
        success = True
        while success:
            cv.imwrite(
                f"{personOriginalFolderPath}/{personName}_{format(count, '04d') }.png",
                image,
            )
            success, image = capture.read()
            count += 1
        print("Images Generated!")

        directory = os.fsencode(personOriginalFolderPath)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            os.system(f"mogrify -rotate -180 {personOriginalFolderPath}/{filename}")
        print("Images Rotated!")

        rc = subprocess.call("model/mtcnnCaller.sh")

        print("Images Cropped!")

        print(rc)


def separateSamples(samplePercent):
    for personFolder in os.listdir(croppedFolderPath):
        if os.path.isfile(f"{croppedFolderPath}{personFolder}"):
            continue

        files = [file for file in os.listdir(f"{croppedFolderPath}{personFolder}/")]
        series = pd.Series(files)
        valSeries = series.sample(frac=samplePercent)
        trainSeries = series.drop(valSeries.index)
        if not os.path.exists(f"{croppedFolderPath}{personFolder}/val/"):
            os.mkdir(f"{croppedFolderPath}{personFolder}/val/")

        if not os.path.exists(f"{croppedFolderPath}{personFolder}/train/"):
            os.mkdir(f"{croppedFolderPath}{personFolder}/train/")

        for file in valSeries:
            if os.path.isfile(f"{croppedFolderPath}{personFolder}/{file}"):
                shutil.move(
                    f"{croppedFolderPath}{personFolder}/{file}",
                    f"{croppedFolderPath}{personFolder}/val/{file}",
                )
        for file in trainSeries:
            if os.path.isfile(f"{croppedFolderPath}{personFolder}/{file}"):
                shutil.move(
                    f"{croppedFolderPath}{personFolder}/{file}",
                    f"{croppedFolderPath}{personFolder}/train/{file}",
                )
