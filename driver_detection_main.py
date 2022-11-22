# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pytesseract # This is the TesseractOCR Python library
# Set Tesseract CMD path to the location of tesseract.exe file
import cv2 as cv
import os

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


DIR=r'Faces'
people=[]

for i in os.listdir(DIR):
   people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = np.load("features.npy")

labels=np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

photo=input("enter the copied path of picture:- ")
img=cv.imread(photo)

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow("Person",gray)
faces_rect= haar_cascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces_rect:
   faces_roi = gray[y:y+h,x:x+h]

   label,confidence= face_recognizer.predict(faces_roi)
   
   print("Label ="+ people[label] +"with a confidence of"+str(confidence))
   
   cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)

   cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
np.load = np_load_old
cv.imshow("Detected Face",img)
cv.waitKey(0)

#
width = 800
height = 400

# load the image, resize it, and convert it to grayscale
image = cv.imread(photo)
image = cv.resize(image, (width, height))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# load the number plate detector
n_plate_detector = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
# detect the number plates in the grayscale image
detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

# loop over the number plate bounding boxes
for (x, y, w, h) in detections:
    # draw a rectangle around the number plate
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv.putText(image, "Number plate detected", (x - 20, y - 10),
                cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

    # extract the number plate from the grayscale image
    number_plate = gray[y:y + h, x:x + w]

cv.imshow("Number plate", number_plate)

cv.imshow("Number plate detection", image)
cv.waitKey(0)
#



pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Read car image and convert color to RGB
carplate_img = cv.imread(photo)
carplate_img_rgb = cv.cvtColor(carplate_img, cv.COLOR_BGR2RGB)
plt.imshow(carplate_img_rgb)
# Import Haar Cascade XML file for Russian car plate numbers
carplate_haar_cascade = cv.CascadeClassifier('haarcascade_russian_plate_number.xml')
# Setup function to detect car plate

def carplate_detect(image):
    carplate_overlay = image.copy() 
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=3)

    for x,y,w,h in carplate_rects: 
        cv.rectangle(carplate_overlay, (x,y), (x+w,y+h), (255,0,0), 5) 
            
        return carplate_overlay

detected_carplate_img = carplate_detect(carplate_img_rgb)
plt.imshow(detected_carplate_img)


# Create function to retrieve only the car plate region itself
def carplate_extract(image):
    
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects: 
            carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
            
    return carplate_img

# Enlarge image for further processing later on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return resized_image

# Display extracted car license plate image
carplate_extract_img = carplate_extract(carplate_img_rgb)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)
plt.imshow(carplate_extract_img);

# Convert image to grayscale
carplate_extract_img_gray = cv.cvtColor(carplate_extract_img, cv.COLOR_RGB2GRAY)
plt.axis('off') 
plt.imshow(carplate_extract_img_gray, cmap = 'gray');

# Apply median blur
carplate_extract_img_gray_blur = cv.medianBlur(carplate_extract_img_gray,3) # kernel size 3
plt.axis('off') 
plt.imshow(carplate_extract_img_gray_blur, cmap = 'gray');

# Display the text extracted from the car plate
print(pytesseract.image_to_string(carplate_extract_img_gray_blur, 
                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

