# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue
from Worker_Camera import cameraInit
from Worker_Processing import processingInit
from threading import Thread

# # # [Library for computer vision]
import numpy as np
import time
import dlib
import cv2
import math

# # # [Library for measurements]
from imutils.video import FPS

# # # [Library for supporting computations]
import numpy as np
import argparse
import datetime
import time
import dlib
import cv2
import copy
import math


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
firstFrame = None


# Zone variables
width = height = None
coordinate = []
zone = alpha = span = None
zoneState = []
zoneCondition = []

# Process variables
detectCamPipe = camDetectPipe = initCameraProcess = None
detectProcessPipe = processDetectPipe = initProcessingProcess = None

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
    global detectCamPipe, camDetectPipe, initCameraProcess
    detectCamPipe, camDetectPipe = Pipe()
    initCameraProcess = Process(target = cameraInit, args = [camDetectPipe])
    initCameraProcess.start()

    detectCamPipe.send(paraInput)
    
    if not detectCamPipe.recv() == "Camera Received":
        print("Error in Pipe multiprocessing: READING CAMERA INPUT")
    
def pointInit():
    global firstFrame, width, height, coordinate
    def click(event, x, y, flags, param):
        global coordinate
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinate.append((x, y))

    width = 550
    firstFrame = detectCamPipe.recv()
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

def processInit():
    global detectProcessPipe, processDetectPipe, initProcessingProcess
    detectProcessPipe, processDetectPipe = Pipe()
    initProcessingProcess = Thread(target = processingInit, args = [processDetectPipe])
    initProcessingProcess.start()

def detectionInit():
    global paraConfig, paraModel, paraNet, paraInput, paraThreshold, paraSkipRate, paraNumLine
    [paraConfig, paraModel, paraNet, paraInput, paraThreshold, paraSkipRate, paraNumLine] = [args.config, args.model, args.net, args.input, args.thr, args.skip, args.line]
    
    processInit()
    netInit()
    frameInit()
    pointInit()
    detectProcessPipe.send(firstFrame)
    detectProcessPipe.send("Frame Sended")
    zoneInit()
    detectProcessPipe.send([zone, zoneState, zoneCondition, span, coordinate])
    detectProcessPipe.send("Zone Sended")
    # detection(detectProcessPipe)


def detection(detectProcessPipe):
    global isCaffe, paraSkipRate, width, height, paraThreshold, personID
    global width, height, coordinate, zone, alpha, span, zoneState
    fps = FPS().start()
    totalFrames = 0
    skip = paraSkipRate
    while True:
        start = time.time()
        frame = detectCamPipe.recv()
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
            print("Fetch time:", time.time() - start)
    
            # Filter detections based on: Confidence and Class
            # start = time.time()
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
            # print("Filter:", time.time() - start)

            # For each rectangle that satisfies the condition, assign a tracker
            for i in np.arange(0, detections.shape[0]):
                boundingBox = detections[i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = boundingBox.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(RGB, rect)

                trackerList.append(tracker)

            # print("Tracker time:", time.time() -start)

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
        # if len(trackerList) < 3:
        #     paraSkipRate = skip * 2
        # else:
        #     paraSkipRate = skip
        totalFrames += 1
        for o in rectList:
            cv2.rectangle(frame, (o[0], o[1], o[2] - o[0], o[3] - o[1]), (0,0,255), 2)

        fps.update()
        cv2.imshow("A", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
        if len(rectList) > 0:
            detectProcessPipe.send([True, rectList, frame])
        else:
            detectProcessPipe.send([False, rectList, frame])
        
        # print(time.time() - start)
    fps.stop()
    print("FPS: {:.2f}".format(fps.fps()))


if __name__ == "__main__":
    detectionInit()
    detection(detectProcessPipe)
