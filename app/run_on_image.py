import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.face_mesh as face_mesh
import numpy as np
import streamlit as st
import json
from PIL import Image
from app.save_on_cloud import saveImage

from app.utils import header_html, sidebar_html
from database.main import get_database

mp_drawing = drawing_utils
mp_face_mesh = face_mesh

input_filepath = "media/input"
output_filepath = "media/output"

DEMO_IMAGE = input_filepath + "/Maeve-The-Boys.png"
DEMO_VIDEO = input_filepath + "/demo.mp4"


def __run_on_image__():
    drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=1, color=drawing_utils.GREEN_COLOR
    )

    st.sidebar.markdown("---")
    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )
    st.markdown("# **Rostos detectados**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input("Número máximo de faces", value=2, min_value=1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider(
        "Confiança mínima para detecção (threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    st.sidebar.markdown("---")
    img_file_buffer = st.sidebar.file_uploader(
        "Faça o upload da imagem aqui", type=["jpg", "jpeg", "png"]
    )

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text("Imagem original")
    st.sidebar.image(image)

    face_count = 0
    imgAtt = {}
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence,
    ) as face_mesh:
        results = face_mesh.process(image)
        out_image = image.copy()
        imgAtt = saveImage(out_image)
        try:
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                    image=out_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                )

                kpi1_text.write(
                    header_html.format(face_count),
                    unsafe_allow_html=True,
                )
            st.subheader("Output image")
            st.image(out_image, use_column_width=True)
        except TypeError:
            pass
    
    mongoConnect = get_database("Images")
    col = mongoConnect.get_collection('Images')
    imgAtt = json.loads(imgAtt)

    col.insert_one({'_id': imgAtt['id'], 'path': imgAtt['path']})
    
