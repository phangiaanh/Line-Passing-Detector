import numpy as np
class TrackableObject:
    def __init__(self, objectID, landmark, lineNum = 2):
        self.objectID = objectID
        self.landmarks = [landmark]

        self.counted = False
        self.up = False

        self.state = np.zeros((lineNum, 2), dtype = bool)
