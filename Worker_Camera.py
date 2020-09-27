# # # [Library for workers]
from multiprocessing import Process, Pipe, Queue

# # # [Library for computer vision]
import numpy as np
import time
import cv2

# Camera variables
capture = isVideo = None
firstFrame = None

def cameraInit(cameraPipe):
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
    camera(cameraPipe)

def camera(cameraPipe):
    global capture, isVideo, firstFrame
    bufferCameraPipe, bufferDetectionPipe = Pipe()
    bufferProcess = Process(target = buffer, args = [bufferDetectionPipe, cameraPipe])
    bufferProcess.start()
    width = 550 
    height = int(width * firstFrame.shape[0] / firstFrame.shape[1])
    lastFrame = cv2.resize(firstFrame, (width, height))
    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame, (width, height))
        if np.sum(abs(frame - lastFrame), axis = None) > 10000:
            bufferCameraPipe.send(frame)
        lastFrame = frame

        if isVideo and frame is None:
            cameraPipe.send("Camera Terminated")
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break


def buffer(bufferCameraPipe, cameraPipe):
    frameQueue = []
    while True:
        if bufferCameraPipe.poll():
             A = bufferCameraPipe.recv()
            frameQueue.append(A)
            cv2.imshow("AAAAA", A)
            if cv2.waitKey(1) == ord('q'):
                break   
        else:
            if len(frameQueue) == 0 or cameraPipe.poll():
                continue
            else:
                start = time.time()
                cameraPipe.send(frameQueue.pop())
                print("Sending in: ", time.time() -start)

        