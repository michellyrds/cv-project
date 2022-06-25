# from sklearn.externals import joblib
import joblib

# create classifier here

clf = "classifier here"


face_detection_model = open("model/face_detection_model.pkl", "wb")

joblib.dump(clf, face_detection_model)

face_detection_model.close()
