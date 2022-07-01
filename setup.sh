mkdir -p ~/.streamlit

apt-get update

apt-get install libsm6 libxext6  -y

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml