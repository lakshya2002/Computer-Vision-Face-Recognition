from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

'''construct the argument parser and parse the arguments'''
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--encodings", required=True,
# 	help="path to serialized db of facial encodings")
# ap.add_argument("-o", "--output", type=str,
# 	help="path to output video")
# ap.add_argument("-y", "--display", type=int, default=1,
# 	help="whether or not to display output frame to screen")
# ap.add_argument("-d", "--detection-method", type=str, default="cnn",
# 	help="face detection model to use: either `hog` or `cnn`")
# args = vars(ap.parse_args())


# load the known faces and embeddings
print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())
data = pickle.loads(open("encodings", "rb").read())
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# cap = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

while True:
    frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1]/float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(
            data["encodings"], encoding)  # return true false
        name = "unknown"
        if True in matches:
            matchedIdx = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdx:
                name = data["names"][i]
                counts[name] = counts.get(name, 0)+1  
            name = max(counts, key =counts.get)    
        names.append(name)


    
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()
