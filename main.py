import streamlit as st

import joblib
import time
from PIL import Image


# # unplick model
model_filepath = "model/face_detection_model.pkl"
# face_detection_model = open(model_filepath, "rb")
# face_detection_clf = joblib.load(face_detection_model)


def detect_face(input):
    pass


def local_css(file_name):
    with open(file_name) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)


def icon_css(icone_name):
    remote_css("https://fonts.googleapis.com/icon?family=Material+Icons")


def icon(icon_name):
    st.markdown(
        '<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True
    )


def load_css(file_name):
    with open(file_name) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def load_images(file_name):
    img = Image.open(file_name)
    return st.image(img, width=300)


def main():
    """Face Detection App
    With Streamlit

    """

    st.title("Face Detection App")
    html_temp = """
        <div style="background-color:blue;padding:10px">
        <h2 style="color:white;text-align:center;">dahora </h2>
        </div>

    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown(
        "<style>" + open("icons.css").read() + "</style>", unsafe_allow_html=True
    )
    st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)

    name = st.file_uploader("Enter image", "Please, upload image here")
    if st.button("Predict"):
        # result = detect_face(image)
        result = list()
        result.append(0)
        if result[0] == 0:
            prediction = "Female"
            img = "images/Maeve-The-Boys.jpg"
        else:
            result[0] == 1
            prediction = "Male"
            img = "images/soldier_boy.jpeg"

        st.success("Name: {} was classified as {}".format(name.title(), prediction))
        load_images(img)


if __name__ == "__main__":
    main()
