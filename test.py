import cv2
from multiprocessing import Pipe, Process

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