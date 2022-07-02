import json
import tempfile
import time
import os

import cv2
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.face_mesh as face_mesh
import streamlit as st

from app.save_on_cloud import saveVideo
from app.utils import header_html, image_resize, sidebar_html
from database.main import get_database

mp_drawing = drawing_utils
mp_face_mesh = face_mesh

input_filepath = "media/input"
output_filepath = "media/output"
output_filename = "/output1.webm"
DEMO_VIDEO = input_filepath + "/demo.mp4"


def __run_on_video__():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    use_webcam = st.sidebar.button("Usar webcam")
    record = st.sidebar.checkbox("Gravar vídeo")

    if record and use_webcam:
        st.checkbox("Gravando...", value=True)

    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )

    max_faces = st.sidebar.number_input("Número máximo de faces", value=5, min_value=1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider(
        "Confiança mínima para detecção",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    tracking_confidence = st.sidebar.slider(
        "Confiança mínima para tracking",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    st.sidebar.markdown("---")

    st.markdown("## Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Faça o upload do vídeo aqui", type=["mp4", "mov", "avi", "asf", "m4v"]
    )
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        video = cv2.VideoCapture(tffile.name)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))

    # Recording
    codec = cv2.VideoWriter_fourcc(*"vp80")
    out = cv2.VideoWriter(
        output_filepath + output_filename, codec, fps_input, (width, height)
    )

    save_video = st.sidebar.checkbox("Salvar output")

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)

    fps, i = 0, 0
    drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=1, color=drawing_utils.GREEN_COLOR
    )

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Faces Detectadas**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Largura da Imagem**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    ) as face_mesh:
        prevTime = 0

        while video.isOpened():
            i += 1
            ret, frame = video.read()

            if not ret:
                break

            results = face_mesh.process(frame)
            frame.flags.writeable = True

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

                # FPS counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if save_video:
                    out.write(frame)

                kpi1_text.write(
                    header_html.format(int(fps)),
                    unsafe_allow_html=True,
                )
                kpi2_text.write(
                    header_html.format(face_count),
                    unsafe_allow_html=True,
                )
                kpi3_text.write(
                    header_html.format(width),
                    unsafe_allow_html=True,
                )

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels="BGR", use_column_width=True)

        video.release()
        out.release()

    if save_video:
        st.text("Vídeo processado")
        output_video = open(output_filepath + output_filename, "rb")

        try:
            videoAtt = saveVideo(output_video)
            videoAtt = json.loads(videoAtt)
            link = "[Vídeo online]({})".format(videoAtt["path"])
            st.markdown(link, unsafe_allow_html=True)

            mongoConnect = get_database("Videos")
            mongoConnect.get_collection("OriginalVideos").insert_one(
                {"_id": videoAtt["id"], "path": videoAtt["path"]}
            )
        except Exception as e:
            print(e)

        # out_bytes = output_video.read()
        # st.video(out_bytes, format="video/webm")
