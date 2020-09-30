from Object import TrackableObject
from imutils.video import VideoStream, FPS
from Tracker import Tracker
from pymongo import MongoClient

import pymongo
from PIL import Image
from bson.binary import Binary
import pickle 
from io import StringIO, BytesIO
from base64 import b64encode
from paho.mqtt import client as mqtt_client
import json
from multiprocessing import Pipe, Process
from threading import Thread
# from test import a,b


import numpy as np
import argparse
import imutils
import datetime
import time
import pytz
import dlib
import cv2
import copy
import math

# With Caffe model:
#     python3 LinePassing.py --config MobileNet/MobileNetSSD_deploy.prototxt --model MobileNet/MobileNetSSD_deploy.caffemodel --net caffe --input /home/ubuntu/Desktop/Video_Campus.mp4 --line 3 --skip 5
# With Tensorflow model:
#     python3 LinePassing.py --config ssd_inception_v2_coco_2018_01_28/graph.pbtxt --model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb  --input /home/ubuntu/Desktop/Video_Campus.mp4 --line 3 --skip 5


# Parser Initializing
parser = argparse.ArgumentParser()
parser.add_argument("--config", required = True,
    help = "Path to [caffe tensorflow] config file")
parser.add_argument("--model", required = True,
    help = "Path to [caffe tensorflow] model file")
parser.add_argument("--net", default = "tensorflow",
    help = "Type of net [caffe tensorflow]")
parser.add_argument("--input",
    help = "Path to video file")
parser.add_argument("--thr", type = float, default = 0.3,
    help = "Threshold")
parser.add_argument("--skip", type = int, default = 10,
    help = "Number of frames skipped")
parser.add_argument("--line", default = 2)
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
if len(args.input) > 2:
    print("[INFO] Opening Video File")
    capture = cv2.VideoCapture(args.input)
    isVideo = True
else:
    print("[INFO] Starting Video Stream")
    capture = cv2.VideoCapture(int(args.input))
    isVideo = False


# Initialize tracking objects
centroidTracker = Tracker(maxDisappeared = 20, maxDistance = 70)
trackableObjects = {}
trackerList = []


# Setup MongoDB
# cluster = MongoClient("mongodb://PGA:111199@cluster0-shard-00-00.gjysk.mongodb.net:27017,cluster0-shard-00-01.gjysk.mongodb.net:27017,cluster0-shard-00-02.gjysk.mongodb.net:27017/<dbname>?ssl=true&replicaSet=atlas-8sqsyx-shard-0&authSource=admin&retryWrites=true&w=majority")
# db = cluster["test"]
# collection = db["test"]


# MQTT Configuration
broker = 'broker.emqx.io'
port = 1883
topic = "hmh-pga-internship-2020"
client_id = '111199'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to ", broker, " port: ", port)
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
client = connect_mqtt()
client.loop_start()



# Set dimension
width = 550
skip = args.skip
coordinate = []
arrows = []



# Set zone
# Click function for visualization
def click(event, x, y, flags, param):
    global coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinate.append((x, y))

_, firstFrame = capture.read()
height = int(width * firstFrame.shape[0] / firstFrame.shape[1])
firstFrame = cv2.resize(firstFrame, (width, height))

cv2.imshow("Zone", firstFrame)
cv2.setMouseCallback("Zone", click)

numLine = int(args.line)
lock = False
while True:
    if len(coordinate) > 0 and not lock:
        i = len(coordinate)
        cv2.circle(firstFrame, (coordinate[i - 1][0], coordinate[i - 1][1]), 4, (255, 255, 255), -1)
        if i % 2 == 0:
            cv2.line(firstFrame, (coordinate[i - 1][0], coordinate[i - 1][1]), (coordinate[i - 2][0], coordinate[i - 2][1]), (255, 255, 255), 1)
        if i == 2 * numLine:
            cv2.setMouseCallback("Zone", lambda *args: None)

    cv2.imshow("Zone", firstFrame)
    if cv2.waitKey(1) == ord('d'):
        cv2.destroyWindow("Zone")
        break

lock = False
coordinate = np.array(coordinate)



# Create zone depending on direction
zone = np.zeros((numLine, 4, 3), dtype = float)
alpha = np.zeros((numLine, 1), dtype = float).flatten()
span = np.zeros((numLine, 1), dtype = float).flatten()
zoneState = []
zoneCondition = []


# Create equations of zones
for i in range(0, int(len(coordinate) // 2)):
    alpha[i] = (coordinate[2 * i, 1] - coordinate[2 * i + 1, 1]) / (coordinate[2 * i, 0] - coordinate[2 * i + 1, 0])
    if abs(coordinate[2 * i, 0] - coordinate[2 * i + 1, 0]) > 2 * abs(coordinate[2 * i, 1] - coordinate[2 * i + 1, 1]):
        span[i] = int(20 * math.sqrt((coordinate[2 * i, 0] - coordinate[2 * i + 1, 0])**2 + (coordinate[2 * i, 1] - coordinate[2 * i + 1, 1])**2) / abs(coordinate[2 * i, 0] - coordinate[2 * i + 1, 0]))
        if coordinate[2 * i, 0] < coordinate[2 * i + 1, 0]:
            zone[i] = np.array([[-alpha[i], 1, -coordinate[2 * i, 1] + span[i] + alpha[i] * coordinate[2 * i, 0]], [-alpha[i], 1, -coordinate[2 * i, 1] - span[i] + alpha[i] * coordinate[2 * i, 0]], [1, 0, -coordinate[2 * i, 0]], [1, 0, -coordinate[2 * i + 1, 0]]])
        else:
            zone[i] = np.array([[-alpha[i], 1, -coordinate[2 * i, 1] + span[i] + alpha[i] * coordinate[2 * i, 0]], [-alpha[i], 1, -coordinate[2 * i, 1] - span[i] + alpha[i] * coordinate[2 * i, 0]], [1, 0, -coordinate[2 * i + 1, 0]], [1, 0, -coordinate[2 * i, 0]]])
        
        zoneState.append(True)
        partCondition = np.array([[[-zone[i, 1, 0], 0, 0, -zone[i, 1, 1], -zone[i, 1, 2] - span[i]], [0, 0, -zone[i, 1, 0], -zone[i, 1, 1], -zone[i, 1, 2] - span[i]]], [[zone[i, 1, 0], 0, 0, zone[i, 1, 1], zone[i, 1, 2]], [0, 0, zone[i, 1, 0], zone[i, 1, 1], zone[i, 1, 2]]]], dtype = float)
        zoneCondition.append(partCondition)
    else:
        span[i] = int(20 * math.sqrt((coordinate[2 * i, 0] - coordinate[2 * i + 1, 0])**2 + (coordinate[2 * i, 1] - coordinate[2 * i + 1, 1])**2) / abs(coordinate[2 * i, 1] - coordinate[2 * i + 1, 1]))
        if coordinate[2 * i, 1] < coordinate[2 * i + 1, 1]:
            zone[i] = np.array([[0, 1, -coordinate[2 * i, 1]], [0, 1, -coordinate[2 * i + 1, 1]], [alpha[i], -1, coordinate[2 * i, 1] - alpha[i] * (coordinate[2 * i, 0] - span[i])] / alpha[i], [alpha[i], -1, coordinate[2 * i, 1] - alpha[i] * (coordinate[2 * i, 0] + span[i])] / alpha[i]])
        else:
            zone[i] = np.array([[0, 1, -coordinate[2 * i + 1, 1]], [0, 1, -coordinate[2 * i, 1]], [alpha[i], -1, coordinate[2 * i, 1] - alpha[i] * (coordinate[2 * i, 0] - span[i])] / alpha[i], [alpha[i], -1, coordinate[2 * i, 1] - alpha[i] * (coordinate[2 * i, 0] + span[i])] / alpha[i]])

        zoneState.append(False)
        partCondition = np.array([[[-zone[i, 2, 0], -zone[i, 2, 1], 0, 0, -zone[i, 1, 2] - span[i]], [-zone[i, 2, 0], 0, 0, -zone[i, 2, 1], -zone[i, 1, 2] - span[i]]], [[0, zone[i, 3, 1], zone[i, 3, 0], 0, zone[i, 3, 2] + span[i]], [0, 0, zone[i, 3, 0], zone[i, 3, 1], zone[i, 3, 2] + span[i]]]])
        zoneCondition.append(partCondition)

zoneCondition = np.array(zoneCondition)

for i in range(0, len(zoneState)):
    xCenter = int((coordinate[2 * i, 0] + coordinate[2 * i + 1, 0] ) / 2)
    yCenter = int((coordinate[2 * i, 1] + coordinate[2 * i + 1, 1] ) / 2)
    if zoneState[i]:
        arrows.append([(xCenter, yCenter - 20), (xCenter, yCenter + 20)])
    else:
        arrows.append([(xCenter - 20, yCenter), (xCenter + 20, yCenter)])

arrows = np.array(arrows)
print(arrows)
# Some output parameters
totalFrames = 0
totalDown = 0
totalUp = 0
totalLeft = 0
totalRight = 0
totalSide = np.zeros((len(zoneState), 2), dtype = int)
fps = FPS().start()

pipe1, pipe2 = Pipe()
def a(pipe2):
    id = 0
    paraInput = pipe2.recv()
    if len(paraInput) > 2:
        capture = cv2.VideoCapture(paraInput)
    else:
        capture = cv2.VideoCapture(int(paraInput))
    # capture = cv2.VideoCapture(0)
    pi, bi = Pipe()
    B = Process(target = b, args = [bi, pipe2])
    # B.start()   
    while True:
        _, frame = capture.read()
        pipe2.send([id, frame])
        # pi.send([id, frame])
        id += 1


def b(e, f):
    frameQueue = []
    while True:
        frame = e.recv()
        frameQueue.append(frame)
        f.send(frameQueue.pop(0))

capture.release()
A = Process(target=a, args=[pipe2])
A.start()
pipe1.send(args.input)
lastID = -1
begin = time.time()
x = 0
while True:
    start = time.time()
    # Get frame
    # _, frame = capture.read()
    id, frame = pipe1.recv()
    x += 1
    # Check video end
    if isVideo and frame is None:
        A.terminate()
        break
    # if time.time() - begin > 20:
    #     print(x)
    #     exit()

    # Resize and Convert frame for dlib
    frame = cv2.resize(frame, (width, height))
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize list of rectangles in each frame
    rectList = []

    if totalFrames % skip == 0:
        trackerList = []

        # Get detections
        if isCaffe:
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), (127.5, 127.5, 127.5), False)
        else:
            blob = cv2.dnn.blobFromImage(frame, size = (width, height))
    
        net.setInput(blob)
        detections = net.forward()
        print("Time:", time.time() - start)

        # Filter detections based on: Confidence and Class
        detections = np.array(detections[0, 0])
        detections = detections[detections[:, 2] > args.thr]
        detections = detections[detections[:, 1] == personID]

        # Filter detections based on: Zone
        zoneCon = np.zeros((len(detections), 1), dtype = bool).flatten()
        for i in range(0, len(zoneState)):
            uExCon = np.logical_and(zone[i, 0, 0] * detections[:, 3] * width + zone[i, 0, 1] * detections[:, 6] * height + zone[i, 0, 2] < 0, zone[i, 0, 0] * detections[:, 5] * width + zone[i, 0, 1] * detections[:, 6] * height + zone[i, 0, 2] < 0)
            dExCon = np.logical_and(zone[i, 1, 0] * detections[:, 3] * width + zone[i, 1, 1] * detections[:, 4] * height + zone[i, 1, 2] > 0, zone[i, 1, 0] * detections[:, 5] * width + zone[i, 1, 1] * detections[:, 4] * height + zone[i, 1, 2] > 0)
            lExCon = np.logical_and(zone[i, 2, 0] * detections[:, 5] * width + zone[i, 2, 1] * detections[:, 4] * height + zone[i, 2, 2] < 0, zone[i, 2, 0] * detections[:, 5] * width + zone[i, 2, 1] * detections[:, 6] * height + zone[i, 2, 2] < 0)
            rExCon = np.logical_and(zone[i, 3, 0] * detections[:, 3] * width + zone[i, 3, 1] * detections[:, 4] * height + zone[i, 3, 2] > 0, zone[i, 3, 0] * detections[:, 3] * width + zone[i, 3, 1] * detections[:, 6] * height + zone[i, 3, 2] > 0)
            if zoneState[i]:
                minCoor = min(coordinate[2 * i, 1], coordinate[2 * i + 1, 1])
                maxCoor = max(coordinate[2 * i, 1], coordinate[2 * i + 1, 1])
                limitCon = np.logical_or(detections[:, 6] * height < minCoor - span[i], detections[:, 4] * height > maxCoor + span[i])
            else:
                minCoor = min(coordinate[2 * i, 0], coordinate[2 * i + 1, 0])
                maxCoor = max(coordinate[2 * i, 0], coordinate[2 * i + 1, 0])
                limitCon = np.logical_or(detections[:, 5] * width < minCoor - span[i], detections[:, 3] * width > maxCoor + span[i])
                
            InCon = np.logical_not(np.logical_or.reduce([uExCon, dExCon, lExCon, rExCon, limitCon]))
            zoneCon = np.logical_or(zoneCon, InCon)
        detections = detections[zoneCon]            


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
            tracker.update(frame)
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
    for i in range(0, len(zoneState)):
        if zoneState[i]:
            color = (77, 77, 255)
            cv2.putText(frame, str( 2 * 0), (arrows[i, 0, 0] + 8, arrows[i, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2)
            cv2.putText(frame, str(2 * 0 + 1), (arrows[i, 1, 0] + 8, arrows[i, 1, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2)
        else:
            color = (255, 77, 77)
            cv2.putText(frame, str(2 * 0), (arrows[i, 0, 0], arrows[i, 0, 1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2)
            cv2.putText(frame, str(2 * 0 + 1), (arrows[i, 1, 0], arrows[i, 1, 1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2)
        cv2.circle(frame, (coordinate[2 * i, 0], coordinate[2 * i, 1]), 4, color, -1)
        cv2.circle(frame, (coordinate[2 * i + 1, 0], coordinate[2 * i + 1, 1]), 4, color, -1)
        cv2.line(frame, (coordinate[2 * i, 0], coordinate[2 * i, 1]), (coordinate[2 * i + 1, 0], coordinate[2 * i + 1, 1]), color, 2)
        cv2.arrowedLine(frame, (*arrows[i, 0], ), (*arrows[i, 1], ), color, 2, tipLength = 0.2)
        cv2.arrowedLine(frame, (*arrows[i, 1], ), (*arrows[i, 0], ), color, 2, tipLength = 0.2)
        
        
        # if zoneState[i]:
        #     cv2.line(frame, (coordinate[2 * i, 0], coordinate[2 * i, 1] + int(span[i])), (coordinate[2 * i + 1, 0], coordinate[2 * i + 1, 1] + int(span[i])), (255, 255, 255), 2)
        #     cv2.line(frame, (coordinate[2 * i, 0], coordinate[2 * i, 1] - int(span[i])), (coordinate[2 * i + 1, 0], coordinate[2 * i + 1, 1] - int(span[i])), (255, 255, 255), 2)
        # else:
        #     cv2.line(frame, (coordinate[2 * i, 0] + int(span[i]), coordinate[2 * i, 1]), (coordinate[2 * i + 1, 0] + int(span[i]), coordinate[2 * i + 1, 1]), (255, 255, 255), 2)
        #     cv2.line(frame, (coordinate[2 * i, 0] - int(span[i]), coordinate[2 * i, 1]), (coordinate[2 * i + 1, 0] - int(span[i]), coordinate[2 * i + 1, 1]), (255, 255, 255), 2)


    copyFrame = copy.copy(frame)
    # Check object conditions
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        to = trackableObjects.get(objectID, None)

		# Create a new one if there is None
        if to is None:
            to = TrackableObject(objectID, centroid, len(zoneState))
            formatCentroid = np.append(centroid[0:4], [1])
            newState = np.logical_and.reduce(np.dot(zoneCondition, formatCentroid) > 0, axis = -1)
            to.state = newState

		# Checking if the conditions are enough to determine passing-line object
        else:
            firstPos = to.landmarks[0]
            formatCentroid = np.append(centroid[0:4], [1])
            # start = time.time()
            newState = np.logical_and.reduce(np.dot(zoneCondition, formatCentroid) > 0, axis = -1)
            
            for i in range(0, len(zoneState)):
                if not np.logical_xor(newState[i, 0], newState[i, 1]):
                    newState[i] = to.state[i]
            
            placeMap = np.logical_and(np.logical_xor(to.state, newState), newState)
            placeMap = np.where(placeMap == True)
            
            if np.logical_or.reduce(np.logical_xor(to.state, newState), axis = None):
                if True in newState[:]:
                    lock = True
                    
                    
            to.state = newState

            # Determine direction
            copyCopyFrame = copy.copy(copyFrame)
            if lock:
                lock = False
                line = int(placeMap[0])
                direction = int(placeMap[1])
                to.place = direction
                totalSide[line, direction] += 1
                if zoneState[line]:
                    if direction == 0:
                        totalUp += 1
                        to.direction = "UP"
                    else:
                        totalDown += 1
                        to.direction = "DOWN"
                else:
                    if direction == 0:
                        totalLeft += 1
                        to.direction = "LEFT"
                    else:
                        totalRight += 1
                        to.direction = "RIGHT"

                copyCopyFrame = cv2.cvtColor(copyCopyFrame, cv2.COLOR_BGR2RGB)
                cv2.rectangle(copyCopyFrame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 255, 255), 2)
                
                
                evidence = Image.fromarray(copyCopyFrame, 'RGB')
                a = b64encode(evidence.tobytes())
                buffer = BytesIO()
                evidence.save(buffer, format = "JPEG")
                base64Image = str(b64encode(buffer.getvalue()))
                
                start = time.time()
                now = datetime.datetime.utcnow()
                now = now.replace(tzinfo = pytz.UTC)
                message = json.dumps({"usr": "hoanghm2","cam_id": "SV10", "line": line, "direction": to.place ,"time": now.isoformat(), "evidence": base64Image})
                result = client.publish(topic, message)
                
                

            
            if not to.direction:
                cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (0, 255, 0), 2)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            else:
                if to.direction == "UP":
                    color = (0, 0, 255)
                elif to.direction == "DOWN":
                    color = (0, 255, 255)
                elif to.direction == "LEFT":
                    color = (128, 255, 128)
                else:
                    color = (255, 128, 128)
                cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), color, 2)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

        trackableObjects[objectID] = to

    text = "{}: {}".format("Area", str(totalSide.flatten()))
    cv2.putText(frame, text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# Display info on frame
    # info = [
	# 	("Up", totalUp),
	# 	("Down", totalDown),
    #     ("Left", totalLeft),    
    #     ("Right", totalRight)
    # ]

    # for (i, (k, v)) in enumerate(info):
    #     text = "{}: {}".format(k, v)
    #     cv2.putText(frame, text, (10, height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    totalFrames += 1
    fps.update()
    cv2.imshow("frame", frame)
    # print(time.time()-start)
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print(totalUp, totalDown)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
