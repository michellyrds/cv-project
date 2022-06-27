from logging.config import valid_ident
from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
from InceptionResnetV1 import inception_resnet_v1
from PIL import Image
from pdb import set_trace
import time
import copy
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import io, transform
from tqdm import trange, tqdm
import csv
import glob
import dlib
import pandas as pd
import numpy as np
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

def generateCroppedImagesFromVideo(videoFilePath, personName = "John_Doe", imagesPath="./data/", samplePercent=0.25):
    originalFolderPath = imagesPath + originalFolder;
    croppedFolderPath = imagesPath + croppedFolder;
    personFolderPath = imagesPath + originalFolder + personName;

    capture = cv.VideoCapture(videoFilePath)

    if(capture.isOpened()):
        success,image = capture.read()
        count = 0
        success = True
        while success:
            cv.imwrite(f"{personFolderPath}/{personName}_{format(count, '04d') }.png", image)
            success,image = capture.read()
            count += 1
        print("Images Generated!")

        directory = os.fsencode(personFolderPath)
            
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            os.system(f'mogrify -rotate -90 {personFolderPath}/{filename}')
        print("Images Rotated!")

        os.system(f'autocrop -i {personFolderPath} -o {personFolderPath}160 -w 720 -H 720 --facePercent 80')
        print("Images Cropped!")

        rc = subprocess.call("./mtcnnCaller.sh")
        print(rc)

def separateSamples(samplePercent): 
    for personFolder in os.listdir(croppedFolderPath):
        files = [file for file in os.listdir(f'{croppedFolderPath}{personFolder}/')]
        series = pd.Series(files)
        valSeries = series.sample(frac=samplePercent)
        trainSeries =  series.drop(valSeries.index)
        os.mkdir(f'{croppedFolderPath}{personFolder}/val/')
        os.mkdir(f'{croppedFolderPath}{personFolder}/train/')
        for file in valSeries: 
            shutil.move(f'{croppedFolderPath}{personFolder}/{file}', f'{croppedFolderPath}{personFolder}/val/{file}')
        for file in trainSeries: 
            shutil.move(f'{croppedFolderPath}{personFolder}/{file}', f'{croppedFolderPath}{personFolder}/train/{file}')

          