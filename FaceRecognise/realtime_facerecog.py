import re
from deepface import DeepFace

#print(DeepFace.analyze('chikku.jpeg'))

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"] #default vgg-face for face verification and recognition

detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]

""" img = DeepFace.detectFace(“img1.jpg”, detector_backend = detectors[4]) """

""" details = DeepFace.verify('chikku.jpeg', 'unknown.jpeg', model_name = models[1])

for i in details:
    print(i,':', details[i], '\n') """

""" recognition = DeepFace.find('unknown.jpeg', db_path = "Images/")
for i in recognition:
    print(i,':', recognition[i], '\n')
 """
DeepFace.stream(db_path = "Images/", model_name = models[-1], detector_backend = detectors[1])
""" print(DeepFace.verify('chikku.jpeg', 'unknown.jpeg'), model )  """