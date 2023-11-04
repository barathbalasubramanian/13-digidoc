from roboflow import Roboflow
import cv2 
rf = Roboflow(api_key="Bjk1PrXgPvle6DAKJAnD")
project = rf.workspace().project("mribraintumordetection")
model = project.version(1).model
model.predict("download.jpg", confidence=40, overlap=30).save("prediction.jpg")
image = cv2.imread('prediction.jpg')
image = cv2.resize(image , (300,300))
cv2.imshow('prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()