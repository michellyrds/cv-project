from .utils.InceptionResnetV1 import InceptionResnetV1
import os
import pandas as pd
import os
import cv2 as cv
import subprocess
import shutil
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch import cuda

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


def separateSamples(samplePercent = 0.25):
    for personFolder in os.listdir(croppedFolderPath):
        if os.path.isfile(f"{croppedFolderPath}{personFolder}") or personFolder == "val" or personFolder == "train":
            continue
        if not os.path.exists(f"{croppedFolderPath}val/"):
            os.mkdir(f"{croppedFolderPath}val/")
        if not os.path.exists(f"{croppedFolderPath}train/"):
            os.mkdir(f"{croppedFolderPath}train/")

        files = [file for file in os.listdir(f"{croppedFolderPath}{personFolder}/")]
        series = pd.Series(files)
        valSeries = series.sample(frac=samplePercent)
        trainSeries = series.drop(valSeries.index)

        if not os.path.exists(f"{croppedFolderPath}val/{personFolder}/"):
            os.mkdir(f"{croppedFolderPath}val/{personFolder}")

        if not os.path.exists(f"{croppedFolderPath}train/{personFolder}"):
            os.mkdir(f"{croppedFolderPath}train/{personFolder}")

        for file in valSeries:
            if os.path.isfile(f"{croppedFolderPath}{personFolder}/{file}"):
                shutil.move(
                    f"{croppedFolderPath}{personFolder}/{file}",
                    f"{croppedFolderPath}val/{personFolder}/{file}",
                )
        for file in trainSeries:
            if os.path.isfile(f"{croppedFolderPath}{personFolder}/{file}"):
                shutil.move(
                    f"{croppedFolderPath}{personFolder}/{file}",
                    f"{croppedFolderPath}train/{personFolder}/{file}",
                )
        shutil.rmtree(f"{croppedFolderPath}{personFolder}")


def augmentData():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(croppedFolderPath, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x],
                                                batch_size=8, 
                                                shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def beginTraining(dataloaders, dataset_sizes, class_names):
    device = cuda.is_available()
    print('Running on device: {}'.format(device))
    model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes = len(class_names))
    print(model_ft)