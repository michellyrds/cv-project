import model.detection as detection


def runDetection():
    detection.generate_paths("media/input/")
    detection.separateSamples(samplePercent=0.3)

    [dataloaders, dataset_sizes, class_names] = detection.augmentData()
    detection.beginTraining("Model_Mixed",dataloaders, dataset_sizes, class_names)


if __name__ == "__main__":
    runDetection()
