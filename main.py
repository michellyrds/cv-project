import streamlit as st

from app.about_app import __about_app__
from app.run_on_image import __run_on_image__
from app.run_on_video import __run_on_video__
from app.utils import sidebar_html
import model.detection as detection


def main():
    """
    Face Detection App with Streamlit

    """
    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Opções")
    st.sidebar.subheader("Parâmetros")

    about_app, run_on_image, run_on_video = (
        "Sobre o app",
        "Rodar em imagem",
        "Rodar em vídeo",
    )

    app_mode = st.sidebar.selectbox(
        "Selecione uma opção", [about_app, run_on_image, run_on_video]
    )

    if app_mode == about_app:
        __about_app__()

    elif app_mode == run_on_video:
        __run_on_video__()

    elif app_mode == run_on_image:
        __run_on_image__()

def runDetection():
    detection.generateCroppedImagesFromVideo('./media/input/Ll-2.mp4')

if __name__ == "__main__":
    #main()
    print("Running Detection")
    runDetection()
    print("AAA Detection")
