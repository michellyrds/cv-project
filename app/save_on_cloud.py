import requests

def checkWorking():
    f = requests.get("https://ex-machina-turbo.herokuapp.com/")
    #f = requests.get("http://localhost:8000/")
    print(f.text)

def saveVideo(video):
    files = {'video': ('video', video,'video/mpeg')}
    f = requests.post("https://ex-machina-turbo.herokuapp.com/api/uploadvideo", files=files)
    return f.text

def saveImage(image):
    files = {'image': ('imagem', image,'image/png')}
    f = requests.post("https://ex-machina-turbo.herokuapp.com/api/uploadimage", files=files)
    return f.text

# checkWorking()

# video = open('media/input/testeNovo1.mp4', 'rb')
# print(video)
# saveVideo(video)

# image = open('media/input/Maeve-The-Boys.jpg', 'rb')
# print(image)
# saveImage(image)