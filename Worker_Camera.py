# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue
from threading import Thread
from Frame import ProcessingFrame

# # # [Library for computer vision]
import numpy as np
import time
import cv2

# Camera variables
capture = isVideo = None
firstFrame = None

def cameraInit(cameraPipe, endQueue):
    global capture, isVideo, firstFrame
    paraInput = cameraPipe.recv()
    cameraPipe.send("Camera Received")
    if len(paraInput) > 1:
        capture = cv2.VideoCapture(paraInput)
        isVideo = True
    else:
        capture = cv2.VideoCapture(int(paraInput))
        isVideo = False

    _, firstFrame = capture.read()
    cameraPipe.send(firstFrame)
    camera(cameraPipe, endQueue)

def camera(cameraPipe, endQueue):
    global capture, isVideo, firstFrame
    frameID = 0
    width = 550
    height = int(width * firstFrame.shape[0] / firstFrame.shape[1])
    lastFrame = cv2.resize(firstFrame, (width, height))
    while True:
        _, frame = capture.read()
        if isVideo and frame is None:
            endQueue.put("END")
            frame = lastFrame
            

        frame = cv2.resize(frame, (width, height))
        processingFrame = ProcessingFrame(frameID, frame)
        frameID += 1
        if frameID > 1000000:
            frameID = 0
        cameraPipe.send(processingFrame)
        lastFrame = frame
        if not endQueue.empty():
            capture.release()
            break

    print("CAMERA DONE")