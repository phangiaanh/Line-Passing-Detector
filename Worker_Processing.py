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



# # # [Library for supporting computations]
import numpy as np
import argparse
import datetime
import time
import dlib
import cv2
import copy
import math


def processingInit(processingPipe):
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
        ret, newRectList, frame = processingPipe.recv()
        objects = centroidTracker.update(newRectList)
        if not ret:
            continue

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
