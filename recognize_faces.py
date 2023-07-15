'''importing the necessary pacakages'''
import face_recognition
import argparse
import pickle
import cv2
import os
from imutils import paths

# import encode_faces

'''Creating an argument parser:'''
ap = argparse.ArgumentParser()
ap.add_argument("-e","--encodings", required=True, help="path to input directory of faces + images")
ap.add_argument("-i","--image", required=True, help="path to serialised db of  facial encodings")
# ap.add_argument("-d","--detection-method",type=str,default='hog', help="facial detection model to use : either 'hog' or 'cnn'")
args = vars(ap.parse_args())

'''load the known faces and embeddings'''
print("[INFO] : loading encodings....")
data = pickle.loads(open(args["encodings"],"rb").read())
# data = pickle.loads(open("encodings","rb").read())


# imagepaths = list(paths.list_images("dataset"))
# for (i, imagepath) in enumerate(imagepaths):
#     name = imagepath.split(os.path.sep)[1]
#     # image = cv2.imread(imagepath)







'''load the input image and convert into BGR to RGB'''
image = cv2.imread(args["image"])
# image = cv2.imread(imagepath)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

'''detect the (x, y)-coordinates of the bounding boxes corresponding 
to each face in the input image, then compute 
the facial embeddings for each face'''
print("[INFO] : recognizing faces...")
# boxes = face_recognition.face_locations(rgb,model=args["detection-method"])
boxes = face_recognition.face_locations(rgb,model="hog")
encodings = face_recognition.face_encodings(rgb,boxes)

'''initialize the list of names for each face detected'''
names = []

'''loop over the facial embeddings'''
for encoding in encodings:
    matches  = face_recognition.compare_faces(data["encodings"],encoding)   #return true false
    name = "unknown"
    
    ''' find the indexes of all matched faces then initialize a
		dictionary to count the total number of times each face
		was matched'''
    if True in  matches:
        matchedIdx = [i for (i,b) in enumerate(matches) if b]
        counts = {}
        # print(matchedIdx)
        '''loop over the matched indexes and maintain a count for
		each recognized face face'''
        for i in matchedIdx:
            name = data["names"][i]
            counts[name] = counts.get(name,0)+1
            # print(counts)    
        '''determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)'''
        name = max(counts,key =counts.get)    
        # print(name)
    '''update the list of name'''
    names.append(name)  
    # print(names) 

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)     
