apt-get update

apt-get install ffmpeg libsm6 libxext6  -y

mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml