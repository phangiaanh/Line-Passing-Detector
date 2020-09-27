# # # [Library for object tracking]
from Object import TrackableObject
from Tracker import Tracker

# # # [Library for MQTT]
from PIL import Image
from io import BytesIO
from base64 import b64encode
from paho.mqtt import client as mqtt_client
import json

# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue
from Worker_Detection import detectionInit, detection



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

processingPipe, detectionPipe = Pipe()
initDetectionProcess = Process(target = detectionInit, args=[detectionPipe])
initDetectionProcess.start()
processingPipe.send([args.config, args.model, args.net, args.input, args.thr, args.skip, args.line])
if not processingPipe.recv() == "Parameter Received":
    print("Error in Pipe multiprocessing")

frame = processingPipe.recv()
if not processingPipe.recv() == "Frame Sended":
    print("Error in Pipe multiprocessing")

[zone, zoneState, zoneCondition, span, coordinate] = processingPipe.recv()
if not processingPipe.recv() == "Zone Sended":
    print("Error in Pipe multiprocessing")

centroidTracker = Tracker(maxDisappeared = 20, maxDistance = 40)
trackableObjects = {}
totalUp = totalDown = totalLeft = totalRight = 0
lock = False
while True:
    if processingPipe.poll():
        newRectList = processingPipe.recv()
    else:
        continue
    objects = centroidTracker.update(newRectList)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid, len(zoneState))
            formatCentroid = np.append(centroid[0:4], [1])
            newState = np.logical_and.reduce(np.dot(zoneCondition, formatCentroid) > 0, axis = -1)
            to.state = newState

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

            if lock:
                lock = False
                line = int(placeMap[0])
                direction = int(placeMap[1])
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
                print(totalUp, totalDown, totalLeft, totalRight)

        trackableObjects[objectID] = to
        # print(time.time() - start)
