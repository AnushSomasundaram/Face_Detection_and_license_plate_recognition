import cv2 as cv

width = 800
height = 400

# load the image, resize it, and convert it to grayscale
image = cv.imread("/Users/software/Desktop/Project_Code/two_people/Screenshot 2022-11-11 at 9.29.42 AM.png")
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