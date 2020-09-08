from Object import TrackableObject
from imutils.video import VideoStream, FPS
from Tracker import Tracker
from pymongo import MongoClient

import pymongo
from bson.binary import Binary 
import pickle
import gridfs
from PIL import Image

import numpy as np
import argparse
import imutils
import datetime
import time
import dlib 
import cv2
import copy


# Parser Initializing
parser = argparse.ArgumentParser()
parser.add_argument("--config", required = True, 
    help = "Path to [caffe tensorflow] config file")
parser.add_argument("--model", required = True,
    help = "Path to [caffe tensorflow] model file")
parser.add_argument("--net", required = True, default = "tensorflow",
    help = "Type of net [caffe tensorflow]")
parser.add_argument("--input",
    help = "Path to video file")
parser.add_argument("--thr", type = float, default = 0.3,
    help = "Threshold")
parser.add_argument("--skip", type = int, default = 10,
    help = "Number of frames skipped")
parser.add_argument("--place", default = "far")
parser.add_argument("--arms", default = "y")
args = parser.parse_args()

# Create Net and personID
if args.net == "caffe":
    net = cv2.dnn.readNetFromCaffe(args.config, args.model)
    personID = 15
    isCaffe = True
else:
    net = cv2.dnn.readNetFromTensorflow(args.model, args.config)
    personID = 1
    isCaffe = False


# Check input
if args.input:
    print("[INFO] Opening Video File")
    capture = cv2.VideoCapture(args.input)
    isVideo = True
else:
    print("[INFO] Starting Video Stream")
    capture = cv2.VideoCapture(0)
    isVideo = False 

centroidTracker = Tracker(maxDisappeared = 20, maxDistance = 70)
trackableObjects = {}
trackerList = []

# Setup MongoDB
cluster = MongoClient("mongodb://PGA:111199@cluster0-shard-00-00.gjysk.mongodb.net:27017,cluster0-shard-00-01.gjysk.mongodb.net:27017,cluster0-shard-00-02.gjysk.mongodb.net:27017/<dbname>?ssl=true&replicaSet=atlas-8sqsyx-shard-0&authSource=admin&retryWrites=true&w=majority")
db = cluster["test"]
collection = db["test"]
fs = gridfs.GridFS(db)
_id = 0

# Set dimension
width = 550
skip = args.skip

# Set zone
_, firstFrame = capture.read()
height = int(width * firstFrame.shape[0] / firstFrame.shape[1])
firstFrame = cv2.resize(firstFrame, (width, height))

zone = cv2.selectROI("Zone", firstFrame, fromCenter = False, showCrosshair = False)
cv2.destroyWindow("Zone")


# Some output parameters
totalFrames = 0
totalDown = 0
totalUp = 0
totalLeft = 0
fps = FPS().start()


while True:
    # Get frame
    _, frame = capture.read()

    # Check video end
    if isVideo and frame is None:
        break

    # Resize and Convert frame for dlib
    frame = cv2.resize(frame, (width, height))
    copyFrame = copy.copy(frame)

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize list of rectangles in each frame
    rectList = []

    if totalFrames % skip == 0:
        trackerList = []

        # Get detections
        blob = cv2.dnn.blobFromImage(frame, size = (width, height))
        net.setInput(blob)
        detections = net.forward()
        

        # Filter detections based on: Confidence and Class
        detections = np.array(detections[0, 0])
        detections = detections[detections[:, 2] > args.thr]
        detections = detections[detections[:, 1] == personID]

        # Filter detections based on: Zone
        detections = detections[np.logical_not(np.logical_and(detections[:, 4] < zone[1] / height, detections[:, 6] < zone[1] / height))]
        detections = detections[np.logical_not(np.logical_and(detections[:, 3] < zone[0] / width, detections[:, 5] < zone[0] / width))]
        detections = detections[np.logical_not(np.logical_and(detections[:, 4] > (zone[1] + zone[3]) / height, detections[:, 6] > (zone[1] + zone[3]) / height))]
        detections = detections[np.logical_not(np.logical_and(detections[:, 3] > (zone[0] + zone[2]) / width, detections[:, 5] > (zone[0] + zone[2]) / width))]


        # For each rectangle that satisfies the condition, assign a tracker 
        for i in np.arange(0, detections.shape[0]):
            boundingBox = detections[i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = boundingBox.astype("int")
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(RGB, rect)

            trackerList.append(tracker)

    else:
        # Update trackers
        for tracker in trackerList:
            tracker.update(RGB)
            position = tracker.get_position()

            startX = int(position.left())
            startY = int(position.top())
            endX = int(position.right())
            endY = int(position.bottom())

            rectList.append((startX, startY, endX, endY))
        
    # Change skip time 
    if len(trackerList) < 3: 
        skip = args.skip * 2
    else:
        skip = args.skip

    # Update objects
    objects = centroidTracker.update(rectList)


    # Visualize results
    cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (0, 0, 255), 3)
    # for (i, (startX, startY, endX, endY)) in enumerate(rectList):
    #     cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)    


    # Check object conditions
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
        else:
            # print([objectID] + to.centroids)
            firstPos = to.landmarks[0]
            upState = firstPos[3] <= zone[1] + zone[3]
            posDiff = centroid - firstPos
            isVertical = 2 * abs(posDiff[5]) > abs(posDiff[4])
            isNear = args.place == "near"
            isUp = posDiff[3] < -zone[3] / 2 and centroid[3] < zone[1] + zone[3] / 2
            isDown = posDiff[3] > zone[3] / 2 and centroid[3] > zone[1] + zone[3]
            isUp = isUp if isNear else isUp and not upState
            isDown = isDown if isNear else isDown and upState
            isVertical = isVertical if args.arms == "y" else True
            to.landmarks.append(centroid)



            copyCopyFrame = copy.copy(copyFrame)
			# Check to see if the object has been counted or not
            if not to.counted:
                
                cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (0, 255, 0), 2)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)   
                if isVertical:
                    if isUp:
                        totalUp += 1
                        to.counted = True
                        to.up = True
                        
                        copyCopyFrame = cv2.cvtColor(copyCopyFrame, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(copyCopyFrame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 255, 255), 2)
                        im = Image.fromarray(copyCopyFrame)
                        im.save("/home/ubuntu/mongodb images/" + str(_id) + ".jpg")
                        # start = time.time()
                        # evidence = fs.put(open(r'evidence.jpg', 'rb'))

                        # now = datetime.datetime.now()
                        # collection.insert_one({"_id": _id, "time": now.strftime("%d/%m/%Y %H:%M:%S"), "direction": "upState", "evidence": evidence})
                        _id = _id + 1
                        
                        # print(time.time()-start)


                    if isDown:
                        totalDown += 1
                        to.counted = True
                        to.up = False
                        
                        copyCopyFrame = cv2.cvtColor(copyCopyFrame, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(copyCopyFrame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 255, 255), 2)
                        im = Image.fromarray(copyCopyFrame)
                        im.save("/home/ubuntu/mongodb images/" + str(_id) + ".jpg")
                        # start = time.time()
                        # evidence = fs.put(open(r'evidence.jpg', 'rb'))

                        # now = datetime.datetime.now()
                        # collection.insert_one({"_id": _id, "time": now.strftime("%d/%m/%Y %H:%M:%S"), "direction": "down", "evidence": evidence})
                        _id = _id + 1
                        # print(time.time()-start)

                else:
                    if posDiff[0] < -zone[2] / 2 and centroid[0] < zone[0]:
                        totalLeft += 1
                        to.counted = True
                        cv2.rectangle(copyCopyFrame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (0, 255, 0), 2)

            else:
                if to.up:
                    cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (0, 255, 255), 2)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
                else:
                    cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 0, 0), 2)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
        trackableObjects[objectID] = to

        
    
	# construct a tuple of information we will be displaying on the
	# frame
    info = [
		("Up", totalUp),
		("Down", totalDown),
        ("Left", totalLeft)
    ]

	# loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    ########################################

    totalFrames += 1
    fps.update()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print(zone)
print(totalUp, totalDown)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# python3 LinePassing.py --config ssd_inception_v2_coco_2018_01_28/graph.pbtxt --model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --input /home/ubuntu/Desktop/pmh_face_15p.mp4 --place far --arms y --net tensorflow