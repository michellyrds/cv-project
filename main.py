import streamlit as st

import model.detection as detection
from app.about_app import __about_app__
from app.run_on_image import __run_on_image__
from app.run_on_video import __run_on_video__
from app.utils import sidebar_html
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import max as tmax, exp as texp

def main():
    """
    Face Detection App with Streamlit

    """
    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Opcoes")
    st.sidebar.subheader("Parametros")

    about_app, run_on_image, run_on_video = (
        "Sobre o app",
        "Rodar em imagem",
        "Rodar em video",
    )

    app_mode = st.sidebar.selectbox(
        "Selecione uma opcao", [about_app, run_on_image, run_on_video]
    )

    if app_mode == about_app:
        __about_app__()

    elif app_mode == run_on_video:
        __run_on_video__()

    elif app_mode == run_on_image:
        __run_on_image__()


def runDetection():
    detection.generate_paths()
    for vid in ['Michelly']:
        if vid == 'vitor':
            rotate = "-180"
        else:
            rotate = "0"
        detection.generateCroppedImagesFromVideo('./media/input/' + vid + '.mp4', personName=vid, rotation=rotate)
    detection.separateSamples()
    [dataloaders, dataset_sizes, class_names] = detection.augmentData()
    detection.beginTraining(dataloaders, dataset_sizes, class_names)

def testModel(model, device):
    model.eval()
    data = datasets.ImageFolder('./media/input', transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    loader = DataLoader(data, batch_size=1, shuffle=True)
    dataIter = iter(loader)
    images, labels = dataIter.next()
    images, labels = images.to(device), labels.to(device)
    ps = texp(model.forward(images))
    _, predTest = tmax(ps,1) 
    print(ps.float())

if __name__ == "__main__":
    #main()
    detection.generateCroppedImagesFromVideo('./media/input/' + "Michelly" + '.mp4', personName="Michelly", rotation="0")