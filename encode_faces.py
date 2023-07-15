# importing the libraries
from imutils import paths
import face_recognition
import cv2
import imutils
import pickle
import os
import argparse

# Creating an argument parser:
ap = argparse.ArgumentParser()
# ap.add_argument("-i","--dataset", required=True, help="path to input directory of faces + images")
# ap.add_argument("-e","--encodings", required=True, help="path to serialised db of  facial encodings")
# ap.add_argument("-d","--detection-method",type=str,default='cnn', help="facial detection model to use : either 'hog' or 'cnn'")
args = vars(ap.parse_args())


'''path to the input images in our dataset'''
print("[INFO] : quantifying faces.....")
# imagepaths = list(paths.list_images(args["dataset"]))
imagepaths = list(paths.list_images("dataset"))
# print(imagepaths)
# print('\n')

'''initialise the list of known encoding and known names'''
knowneEncoding = []
KnownNames = []

'''loop over the image paths'''
# i---> 0,1,2,....    and  imagepath----->dataset/aastha/1.jpg
for (i, imagepath) in enumerate(imagepaths):
    # extract the person name from image paths
    # print("index : ",i)
    # print ("imagepath : ",imagepath)
    print("[INFO] : processing image {}/{} ".format(i+1, len(imagepaths)))
    name = imagepath.split(os.path.sep)[1]
    # print("Extracted name : ", name)
    # print('\n')
    '''load the input image and convert it into BGR(opencv odering ) to  RGB (dlib odering)'''
    image = cv2.imread(imagepath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    '''detect the (x, y)-coordinates of the bounding boxes
	corresponding to each face in the input image'''
    # boxes = face_recognition.face_locations(rgb,model=args["detection-method"])
    boxes = face_recognition.face_locations(rgb, model="hog")

    '''compute the face encoding for the face'''
    encodings = face_recognition.face_encodings(rgb, boxes)

    '''loop over the encodings'''
    for encoding in encodings:
        knowneEncoding.append(encoding) 
        KnownNames.append(name)
            
'''dump the facial aencodings + names to the disks'''
print("[INFO] : searializing encodings....")
data = {"encodings":knowneEncoding,"names":KnownNames}
# f = open(args["encodings"],"wb")        
f = open("encodings","wb")        
f.write(pickle.dumps(data))
f.close()