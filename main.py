import streamlit as st

import model.detection as detection
from app.about_app import __about_app__
from app.run_on_image import __run_on_image__
from app.run_on_video import __run_on_video__
from app.utils import sidebar_html

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
    #detection.generateCroppedImagesFromVideo('./media/input/video.mp4')
    #detection.separateSamples()
    [dataloaders, dataset_sizes, class_names] = detection.augmentData()
    detection.beginTraining(dataloaders, dataset_sizes, class_names)

if __name__ == "__main__":
    #main()
    print("Running Detection")
    runDetection()
