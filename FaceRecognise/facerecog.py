import face_recognition

knownimage = face_recognition.load_image_file('chikku.jpeg')
unknownimage = face_recognition.load_image_file('unknown.jpeg')

chikku_encode = face_recognition.face_encodings(knownimage)[0]
unknown_encode = face_recognition.face_encodings(unknownimage)[0]

print(face_recognition.compare_faces([chikku_encode], unknown_encode))