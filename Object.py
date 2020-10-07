import numpy as np
class TrackableObject:
    def __init__(self, objectID, landmark, lineNum = 2):
        self.objectID = objectID
        self.landmarks = [landmark]

        self.direction = None
        self.place = None

        self.state = np.zeros((lineNum, 2), dtype = bool)

    def getRectList(self):
        return [self.landmarks[-1], self.direction]