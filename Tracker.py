from scipy.spatial import distance as dist 
from collections import OrderedDict
import numpy as np

class Tracker:
    def __init__(self, maxDisappeared = 40, maxDistance = 40):
        
        # Initialize next object ID in Dictionary
        self.nextObjectID = 0

        # Initialize two dictionaries for current and missing objects
        self.currentObjects = OrderedDict()
        self.disappearedObjects = OrderedDict()

        # Set maximum distance for an object to be recognized again
        # and maximum frame allowed for an object to be disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    
    def register(self, landmarks):
        # Create a new object in dictionary
        self.currentObjects[self.nextObjectID] = landmarks
        self.disappearedObjects[self.nextObjectID] = 0

        self.nextObjectID += 1

    
    def deregister(self, objectID):
        del self.currentObjects[objectID]
        del self.disappearedObjects[objectID]


    
    def update(self, boundingBoxes):
        
        # If there is no bounding boxes, update disappered dictionary
        if len(boundingBoxes) == 0:
            for objectID in list(self.disappearedObjects.keys()):
                self.disappearedObjects[objectID] += 1

                # Delete object when it has disappeared too long
                if self.disappearedObjects[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.currentObjects

        # Format boundingBoxes into Landmarks
        inputLandmarks = np.zeros((len(boundingBoxes), 6), dtype = int)

        for (i, (startX, startY, endX, endY)) in enumerate(boundingBoxes):
            cX = int((startX + endX) / 2)
            cY = int((startY + endY) / 2)

            inputLandmarks[i] = (startX, startY, endX, endY, cX, cY)

        # Register new objects if there is no current object
        if len(self.currentObjects) == 0:
            for i in range(0, len(inputLandmarks)):
                self.register(inputLandmarks[i])

        # Matching current objects with new bounding boxes
        else:
            # Get objects
            objectIDs = np.array(list(self.currentObjects.keys()))
            objectLandmarks = np.array(list(self.currentObjects.values()))


            # Matching strategy will be based on distance map
            distanceMap = dist.cdist(objectLandmarks[:, 4 : 6], inputLandmarks[:, 4 : 6])

            # Create minimum index map
            minimumRows = distanceMap.min(axis = 1).argsort()
            minimumCols = distanceMap.argmin(axis = 1)[minimumRows]

            # Create a record to track examined objects
            usedRows = set()
            usedCols = set()


            # Matching objects
            for (row, col) in zip(minimumRows, minimumCols):

                if row in usedRows or col in usedCols:
                    continue

                if distanceMap[row, col] > self.maxDistance:
                    continue

                # Update new landmarks
                objectID = objectIDs[row]
                self.currentObjects[objectID] = inputLandmarks[col]
                self.disappearedObjects[objectID] = 0

                # Add to used position
                usedRows.add(row)
                usedCols.add(col)


            unusedRows = set(range(0, distanceMap.shape[0])).difference(usedRows)
            unusedCols = set(range(0, distanceMap.shape[1])).difference(usedCols)

            if distanceMap.shape[0] >= distanceMap.shape[1]:
                
                # If we are missing some objects, increase their disappeared rates
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappearedObjects[objectID] += 1

                    if self.disappearedObjects[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputLandmarks[col])


        return self.currentObjects

