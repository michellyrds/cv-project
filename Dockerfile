FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run", "main.py"]

CMD ["main.py"]
