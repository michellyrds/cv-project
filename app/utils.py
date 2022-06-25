
import streamlit as st
import cv2

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

