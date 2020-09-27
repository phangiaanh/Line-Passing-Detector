# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue
from Worker_Camera import cameraInit

# # # [Library for computer vision]
import numpy as np
import time
import dlib
import cv2
import math

# Input parameters
paraConfig = paraModel = paraNet = paraThreshold = None
paraInput = None
paraSkipRate = None
paraNumLine = None

# Neural network variables
net = None
personID = None
isCaffe = None

# Tracking variables
trackerList = []

# Camera variables
detectionPipe = cameraPipe = initCameraProcess = None
firstFrame = None


# Zone variables
width = height = None
coordinate = []
zone = alpha = span = None
zoneState = []
zoneCondition = []


def netInit():
    global net, personID, isCaffe
    global paraConfig, paraModel
    if paraNet == "caffe":
        net = cv2.dnn.readNetFromCaffe(paraConfig, paraModel)
        personID = 15
        isCaffe = True
    else:
        net = cv2.dnn.readNetFromTensorflow(paraModel, paraConfig)
        personID = 1
        isCaffe = False

def frameInit():
    global detectionPipe, cameraPipe, initCameraProcess

    detectionPipe, cameraPipe = Pipe()
    initCameraProcess = Process(target = cameraInit, args = [cameraPipe])
    initCameraProcess.start()

    detectionPipe.send(paraInput)
    
    if not detectionPipe.recv() == "Camera Received":
        print("Error in Pipe multiprocessing: READING CAMERA INPUT")
    

def pointInit():
    global firstFrame, width, height, coordinate
    def click(event, x, y, flags, param):
        global coordinate
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinate.append((x, y))

    width = 550
    firstFrame = detectionPipe.recv()
    height = int(width * firstFrame.shape[0] / firstFrame.shape[1])
    firstFrame = cv2.resize(firstFrame, (width, height))

    cv2.imshow("Zone", firstFrame)
    cv2.setMouseCallback("Zone", click)

    lock = False
    while True:
        if len(coordinate) > 0 and not lock:
            i = len(coordinate)
            cv2.circle(firstFrame, (coordinate[i - 1][0], coordinate[i - 1][1]), 4, (255, 255, 255), -1)
            if i % 2 == 0:
                cv2.line(firstFrame, (coordinate[i - 1][0], coordinate[i - 1][1]), (coordinate[i - 2][0], coordinate[i - 2][1]), (255, 255, 255), 1)
            if i == 2 * paraNumLine:
                cv2.setMouseCallback("Zone", lambda *args: None)

        cv2.imshow("Zone", firstFrame)
        if cv2.waitKey(1) == ord('d'):
            cv2.destroyWindow("Zone")
            break

    coordinate = np.array(coordinate)

def zoneInit():
    global zone, zoneState, zoneCondition
    global alpha, span, coordinate
    global width, height

    zone = np.zeros((paraNumLine, 4, 3), dtype = float)
    alpha = np.zeros((paraNumLine, 1), dtype = float).flatten()
    span = np.zeros((paraNumLine, 1), dtype = float).flatten()

    for i in range(0, int(len(coordinate) / 2)):
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



def detectionInit(detectProcPipe):
    global paraConfig, paraModel, paraNet, paraInput, paraThreshold, paraSkipRate, paraNumLine
    [paraConfig, paraModel, paraNet, paraInput, paraThreshold, paraSkipRate, paraNumLine] = detectProcPipe.recv()
    detectProcPipe.send("Parameter Received")
    netInit()
    frameInit()
    pointInit()
    detectProcPipe.send(firstFrame)
    detectProcPipe.send("Frame Sended")
    zoneInit()
    detectProcPipe.send([zone, zoneState, zoneCondition, span, coordinate])
    detectProcPipe.send("Zone Sended")
    detection(detectProcPipe)
    # detectProcPipe.close()


def detection(detectProcPipe):
    global isCaffe, paraSkipRate, width, height, paraThreshold, personID
    global width, height, coordinate, zone, alpha, span, zoneState
    totalFrames = 0
    skip = paraSkipRate
    while True:
        # frame = detectionPipe.recv()
        if detectionPipe.poll():
            frame = detectionPipe.recv()
        # Resize and Convert frame for dlib
        frame = cv2.resize(frame, (width, height))

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize list of rectangles in each frame
        rectList = []
        if totalFrames % paraSkipRate == 0:
            trackerList = []

            # Get detections
            if isCaffe:
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), (127.5, 127.5, 127.5), False)
            else:
                blob = cv2.dnn.blobFromImage(frame, size = (width, height))

            net.setInput(blob)
            detections = net.forward()


            # Filter detections based on: Confidence and Class
            detections = np.array(detections[0, 0])
            detections = detections[detections[:, 2] > paraThreshold]
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
            paraSkipRate = skip * 2
        else:
            paraSkipRate = skip
        totalFrames += 1
        # for o in rectList:

        #     cv2.rectangle(frame, (o[0], o[1], o[2] - o[0], o[3] - o[1]), (0,0,255), 2)
        cv2.imshow("A", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        detectProcPipe.send(rectList)
        # print(time.time() - start)