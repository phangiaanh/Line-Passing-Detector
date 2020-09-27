from Object import TrackableObject
from imutils.video import VideoStream, FPS
from Tracker import Tracker
from pymongo import MongoClient

import pymongo
from PIL import Image
from bson.binary import Binary
import pickle 
from io import StringIO, BytesIO
from base64 import b64encode
from paho.mqtt import client as mqtt_client
import json


import numpy as np
import argparse
import imutils
import datetime
import time
import dlib
import cv2
import copy
import math

from multiprocessing import Pipe, Process


def getFrame(child):
    capture = cv2.VideoCapture("/home/ubuntu/Desktop/Video_Campus.mp4")
    while True:
        _, frame = capture.read()
        start = time.time()
        child.send(frame)
        print("Sending time: ", time.time() - start)
        if child.recv() == 'q':
            child.close()
            exit()