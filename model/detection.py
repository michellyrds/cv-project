import copy
import os
import pickle
import shutil
import subprocess
import time
from turtle import color

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch import cuda, device
from torch import max as tmax
from torch import nn, optim, set_grad_enabled
from torch import sum as tsum
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils.InceptionResnetV1 import InceptionResnetV1

# USAR ESSA VERSÃO DO TENSORFLOW !pip install tensorflow==1.13.1
originalFolder = "original/"
croppedFolder = "cropped/"
originalFolderPath = ""
croppedFolderPath = ""
personFolderPath = ""


def generate_paths(imagesPath="./data/"):
    global croppedFolderPath 
    croppedFolderPath = imagesPath


def generateCroppedImagesFromVideo(
    videoFilePath, personName="John_Doe", imagesPath="./data/", rotation="0"
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
            os.system(
                f"mogrify -rotate {rotation} {personOriginalFolderPath}/{filename}"
            )
        print("Images Rotated!")

        rc = subprocess.call("model/mtcnnCaller.sh")

        print("Images Cropped!")

        print(rc)


def test_video_orientation(videoFilePath, personName="John_Doe", testPath="./test/"):
    if not os.path.exists(testPath):
        os.mkdir(f"{testPath}")

    capture = cv.VideoCapture(videoFilePath)

    if capture.isOpened():
        success, image = capture.read()
        cv.imwrite(
            f"{testPath}{personName}_orientation_{format(0, '04d') }.png",
            image,
        )


def separateSamples(samplePercent=0.25):
    for personFolder in os.listdir(croppedFolderPath):
        if (
            os.path.isfile(f"{croppedFolderPath}{personFolder}")
            or personFolder == "val"
            or personFolder == "train"
        ):
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
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(croppedFolderPath, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=8, shuffle=True)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def loadModel(modelFilePath="./model/trainedModel.pth"):
    if os.path.isfile(modelFilePath):
        print("Arquivo de modelo encontrado. Pulando o treinamento.")
        file = open(modelFilePath, "rb")

        data = pickle.load(file)

        file.close()
        return data
    else:
        print("Arquivo de modelo não encontrado. Parando...")
        return None


def getPytorchDevice():
    return device("cuda" if cuda.is_available() else "cpu")


def beginTraining(model_name,dataloaders, dataset_sizes, class_names):
    current_device = getPytorchDevice()
    print("Running on device: {}".format(current_device))

    model_ft, layer_list = createModel(class_names)

    model_ft, criterion, optimizer, scheduler = addFinalLayers(
        class_names, model_ft, current_device, layer_list
    )
    model, losses, losses_val = train_model(
        model_name,
        model_ft,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        current_device,
    )
    eval_model(losses, losses_val)


def createModel(class_names):
    model_ft = InceptionResnetV1(
        pretrained="vggface2", classify=False, num_classes=len(class_names)
    )
    print(model_ft)
    layer_list = list(model_ft.children())[-5:]
    print(layer_list)
    model_ft = nn.Sequential(*list(model_ft.children())[:-5])
    for param in model_ft.parameters():
        param.requires_grad = False
    return model_ft, layer_list


def addFinalLayers(class_names, model_ft, device, layer_list):
    model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    model_ft.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=512, bias=False),
        normalize(),
    )
    model_ft.logits = nn.Linear(layer_list[-1].in_features, len(class_names))
    model_ft.softmax = nn.Softmax(dim=1)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)
    # Decay LR by a factor of *gamma* every *step_size* epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def train_model(
    model_name,
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25,
):
    since = time.time()
    FT_losses = []
    FT_losses_val = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = tmax(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                if phase == "train":
                    FT_losses.append(loss.item())
                if phase == "val":
                    FT_losses_val.append(loss.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += tsum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # path = "./model/trainedModel.pth"
    path = "./model/{}.z".format(model_name)
    file = open(path, "wb")
    pickle.dump(model, file)
    file.close()
    # save(model.state_dict(), path)

    return model, FT_losses, FT_losses_val


def eval_model(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.title("FRT Loss During Training")
    plt.plot(train_losses, label="FT loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("FRT Loss During Validation")
    plt.plot(val_losses, label="FT loss", color="orange")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x
