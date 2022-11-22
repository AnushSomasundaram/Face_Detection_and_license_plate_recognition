from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def gray_scale(image):
   input_image = imread(image)
   
   r,g,b=input_image[:,:,0],input_image[:,:,1],input_image[:,:,2]

   gamma=1.04

   r_const,g_const,b_const = 0.3 , 0.59 , 0.11

   grayscale_image = r_const * r**gamma + g_const * g**gamma + b_const * b**gamma

   fig=plt.figure(1)
   img1,img2=fig.add_subplot(121),fig.add_subplot(122)

   img1.imshow(input_image)
   img2.imshow(grayscale_image,cmap=plt.cm.get_cmap("gray"))
   fig.show()
   plt.show()

   
gray_scale("/Users/somradhakrishnan/Google Drive/Anush_projects/linear_algebra/Faces/Jerry Seinfield/5.jpg")
gray_scale("/Users/somradhakrishnan/Google Drive/Anush_projects/linear_algebra/Faces/Madonna/2.jpg")
gray_scale("Faces/Elton John/5.jpg")


