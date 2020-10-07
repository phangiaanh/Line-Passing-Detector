# # #  [Library for visualization]
import cv2
import numpy as np


def visualizeInit(detectPipe, processPipe):
    rectList = np.array([])
    color = (255,255,255)
    while True:
        if  processPipe.poll():
            receivedList = processPipe.recv()
            rectList = np.array(receivedList, dtype = object)  
        else:
            rectList = np.array([])  
        frame = detectPipe.recv()

        if len(rectList) > 0:
            for box in rectList:
                if box[1]:
                    if box[1] == "UP":
                        color = (0, 0, 255)
                    elif box[1] == "DOWN":
                        color = (0, 255, 255)
                    elif box[1] == "LEFT":
                        color = (128, 255, 128)
                    else:
                        color = (255, 128, 128)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (box[0][0], box[0][1]), (box[0][2], box[0][3]), color, 2)

        cv2.imshow("Visualization View", frame)
        if cv2.waitKey(1) == ord('q'):
            break