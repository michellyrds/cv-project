from cv2 import circle
from nbformat import write
import streamlit as st
import cv2
import joblib
import time
from PIL import Image
import numpy as np
import tempfile
import time
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.face_mesh as face_mesh

from app.utils import image_resize, sidebar_html
# # unplick model
model_filepath = "model/face_detection_model.pkl"
# face_detection_model = open(model_filepath, "rb")
# face_detection_clf = joblib.load(face_detection_model)

mp_drawing = drawing_utils
mp_face_mesh = face_mesh
DEMO_IMAGE = 'media/Maeve-The-Boys.jpg'
DEMO_VIDEO = 'media/demo.mp4'

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
    """
    Face Detection App with Streamlit

    """
    st.markdown(
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
        """,
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
        st.markdown("Descrever nossa aplicação aqui")
    
        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )
    elif app_mode == run_on_video:
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        use_webcam = st.sidebar.button('Usar Webcam')
        record = st.sidebar.checkbox("Gravar vídeo")

        if record:
            st.checkbox("Gravando...", value=True)

        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )

        max_faces = st.sidebar.number_input("Número máximo de faces", value=5, min_value=1)
        st.sidebar.markdown("---")
        detection_confidence = st.sidebar.slider('Confiança mínima para detecção (threshold)', min_value=0.0, max_value=1.0, value=0.5)
        tracking_confidence = st.sidebar.slider('Confiança mínima para tracking (threshold)', min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.markdown("---")

        st.markdown("## Output")

        stframe = st.empty()
        video_file_buffer = st.sidebar.file_uploader("Faça o upload do vídeo aqui", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
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

        ## Recording
        # codec = cv2.VideoWriter_fourcc('M', 'J','P','G')
        # codec = cv2.VideoWriter_fourcc('V', 'P','0','9')
        codec = cv2.VideoWriter_fourcc('M', 'P','4','V')
        out = cv2.VideoWriter('media/output1.mp4', codec, fps_input, (width, height))

        st.sidebar.text('Input Video')
        st.sidebar.video(tffile.name)

        fps, i = 0, 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=drawing_utils.GREEN_COLOR)

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

        face_count = 0
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as face_mesh:
                prevTime = 0

                while video.isOpened():
                    i+=1
                    ret, frame = video.read()

                    if not ret:
                        continue
                    
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame)
                    frame.flags.writeable = True
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    face_count=0
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            face_count += 1
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections= mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec = drawing_spec,
                                connection_drawing_spec=drawing_spec)

                        # FPS counter logic
                        currTime = time.time()
                        fps = 1/(currTime - prevTime)
                        prevTime = currTime

                        if record:
                            out.write(frame)

                        kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                        kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                        kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

                        frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
                        frame = image_resize(image=frame, width=640)
                        stframe.image(frame, channels='BGR', use_column_width=True)

                
                st.text("Vídeo processado")
                output_video  = open('media/output1.mp4', 'rb')
                out_bytes = output_video.read()
                st.video(out_bytes)

                video.release()
                out.release()







    elif app_mode == run_on_image:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=drawing_utils.GREEN_COLOR)

        st.sidebar.markdown("---")
        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )
        st.markdown("# **Rostos detectados**")
        kpi1_text = st.markdown("0")

        max_faces = st.sidebar.number_input("Número máximo de faces", value=2, min_value=1)
        st.sidebar.markdown("---")
        detection_confidence = st.sidebar.slider('Confiança mínima para detecção (threshold)', min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.markdown("---")
        img_file_buffer = st.sidebar.file_uploader("Faça o upload da imagem aqui", type=['jpg', 'jpeg', 'png'])

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

        st.sidebar.text('Imagem original')
        st.sidebar.image(image)

        face_count = 0
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence) as face_mesh:
                results = face_mesh.process(image)
                out_image = image.copy()
                try:
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
                        mp_drawing.draw_landmarks(
                            image=out_image,
                            landmark_list=face_landmarks,
                            connections= mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec = drawing_spec)


                        kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                    st.subheader("Output image")
                    st.image(out_image, use_column_width=True)
                except TypeError:
                    pass
            




if __name__ == "__main__":
    main()
