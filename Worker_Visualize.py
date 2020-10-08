# # #  [Library for visualization]
import cv2
import numpy as np


def visualizeInit(processPipe):
    colorTable = {"UP": (0, 0, 255), "DOWN": (0, 255, 255), "LEFT": (128, 255, 128), "RIGHT": (255, 128, 128), "NONE": (0, 255, 0)}
    while True:  
        processingFrame = processPipe.recv()
        
        # Extract info from processingFrame
        frame = processingFrame.image
        height = frame.shape[0]
        rectList = np.array(processingFrame.box)
        rectColor = processingFrame.color
        [totalUp, totalDown, totalLeft, totalRight] = processingFrame.total

        for i in range(0, len(rectList)):
            cv2.rectangle(frame, (rectList[i, 0], rectList[i, 1]), (rectList[i, 2], rectList[i, 3]), colorTable[rectColor[i]], 2)
        info = [
		("UP", totalUp),
		("DOWN", totalDown),
        ("LEFT", totalLeft),    
        ("RIGHT", totalRight)
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorTable[k], 2)
        
        cv2.imshow("Visualization View", frame)
        if cv2.waitKey(1) == ord('q'):
            processPipe.send("END")
            break