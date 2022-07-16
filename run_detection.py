import model.detection as detection


def runDetection():
    detection.generate_paths()
    [dataloaders, dataset_sizes, class_names] = detection.augmentData()
    detection.beginTraining(dataloaders, dataset_sizes, class_names)


if __name__ == "__main__":
    runDetection()
