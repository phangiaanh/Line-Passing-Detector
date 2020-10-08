# # # [Library for object tracking]
from Object import TrackableObject
from Tracker import Tracker

# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue
from Worker_MQTT import mqttInit
from Worker_Visualize import visualizeInit

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
    # Receive information
    frame = processingPipe.recv()
    if not processingPipe.recv() == "Frame Sended":
        print("Error in Pipe multiprocessing")

    [zone, zoneState, zoneCondition, span, coordinate] = processingPipe.recv()
    if not processingPipe.recv() == "Zone Sended":
        print("Error in Pipe multiprocessing")


    # Tracker objects
    centroidTracker = Tracker(maxDisappeared = 20, maxDistance = 40)
    trackableObjects = {}
    totalUp = totalDown = totalLeft = totalRight = 0
    lock = False
    

    # Visualize Process
    processVisualizePipe, visualizeProcessPipe = Pipe()
    visualizeInitProcess = Process(target = visualizeInit, args = [visualizeProcessPipe])
    visualizeInitProcess.start()



    # MQTT Process
    processMQTTPipe, mqttProcessPipe = Pipe()
    mqttInitProcess = Process(target = mqttInit, args = [mqttProcessPipe])
    mqttInitProcess.start()
    
    while True:
        if processVisualizePipe.poll():
            processingPipe.send("END")
            break
        processingFrame = processingPipe.recv()
        ret = processingFrame.isSomeone
        newRectList = processingFrame.box
        updateRect = []
        updateColor = []
        frame = processingFrame.image
        start = time.time()
        objects = centroidTracker.update(newRectList)
        if not ret:
            processingFrame.total = [totalUp, totalDown, totalLeft, totalRight]
            processVisualizePipe.send(processingFrame)
            continue
        
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid, len(zoneState))
                formatCentroid = np.append(centroid[0:4], [1])
                newState = np.logical_and.reduce(   np.dot(zoneCondition, formatCentroid) > 0, axis = -1)
                to.state = newState

            else:
                firstPos = to.landmarks[0]
                to.landmarks.append(centroid)
                formatCentroid = np.append(centroid[0:4], [1])
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
                    processMQTTPipe.send([frame, line, direction, centroid])

                updateRect.append(centroid)
                if to.direction:
                    updateColor.append(to.direction)
                else:
                    updateColor.append("NONE")    

            trackableObjects[objectID] = to

        processingFrame.box = updateRect
        processingFrame.color = updateColor  
        processingFrame.processingTime["Processing"] = time.time() - start
        processingFrame.total = [totalUp, totalDown, totalLeft, totalRight]
        processVisualizePipe.send(processingFrame)
        
        newKeys = list(objects.keys())
        trackableObjects = dict([(key, trackableObjects[key]) for key in newKeys])
        objectList = [value.getRectList() for key, value in trackableObjects.items()]

    print("DONE")