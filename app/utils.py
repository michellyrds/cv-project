
import streamlit as st
import cv2
from PIL import Image

@st.cache()
def image_resize(
    image, width=None, heigth=None, inter=cv2.INTER_AREA
):  # inter: interpolation method
    dim = None
    (h, w) = image.shape[:2]

    if width is None and heigth is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), heigth)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_image = cv2.resize(image, dim, interpolation=inter)

    return resized_image


sidebar_html = (
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
    """
)

header_html = "<h1 style='text-align: center; color:rgb(255, 75, 75);'>{}</h1>"

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


def load_images(file_name, width):
    img = Image.open(file_name)
    return st.image(img, width=width)
