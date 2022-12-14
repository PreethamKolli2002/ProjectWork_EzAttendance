import cv2
import numpy as np
import face_recognition
from PIL import Image

imgBill = face_recognition.load_image_file(r'Images_Basic\bill_gates_1.jpeg')
# imgBill = Image.open(r'Images_Basic\bill_gates_1.jpeg')
imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)
imgBill1 = face_recognition.load_image_file(r'Images_Basic\bill_gates_2.jpeg')
imgBill1 = cv2.cvtColor(imgBill1,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgBill)[0] # Finds the approx location of the face
encodeBill = face_recognition.face_encodings(imgBill)[0] # Encodes the face location. Used for comparing if two image  are identical or not.
cv2.rectangle(imgBill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)
# cv2.circle(imgBill,(faceLoc[3],faceLoc[0]),10,(255,0,0),2)
# cv2.circle(imgBill,(faceLoc[1],faceLoc[2]),10,(0,255,0),2)
# cv2.circle(imgBill,(faceLoc[2]-5,faceLoc[2]+5),10,(0,0,255),2)
# cv2.circle(imgBill,(faceLoc[3]-5,faceLoc[3]+5),10,(255,0,255),2)
# imgCropped1 = imgBill[faceLoc[0]:faceLoc[0]+(faceLoc[3]-faceLoc[0]),faceLoc[3]:faceLoc[3]+faceLoc[1]]
# cv2.imshow('Test',imgBill)
# cv2.waitKey()
print(faceLoc)
# cv2.imwrite(r'C:\Preetham_Kolli\CSE\Python\AI_ML\Face_Recognition\Cropped_Images\bill_crop.jpeg',imgCropped1)
# print(faceLoc)

faceLocTest = face_recognition.face_locations(imgBill1)[0]
encodeBillTest = face_recognition.face_encodings(imgBill1)[0]
cv2.rectangle(imgBill1,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) # Draws a rectangle around face
# imgCropped2 = imgBill1[faceLoc[3]:faceLoc[0],faceLoc[3]+faceLoc[1]:faceLoc[0]+faceLoc[2]]
# cv2.imwrite(r'C:\Preetham_Kolli\CSE\Python\AI_ML\Face_Recognition\Cropped_Images\bill1_crop.jpeg',imgCropped1)
results = face_recognition.compare_faces([encodeBill],encodeBillTest) # Returns true if images are identical
faceDis = face_recognition.face_distance([encodeBill],encodeBillTest) # Checks how identical two images are.
# print(faceDis)

cv2.putText(imgBill,f'{results} {round(faceDis[0],2)}',(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Bill Gates',imgBill)
cv2.imshow('Bill Test',imgBill1)
cv2.waitKey()

# vid = cv2.VideoCapture(0)
#
# while (True):
#
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()
