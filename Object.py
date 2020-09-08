class TrackableObject:
    def __init__(self, objectID, landmark):
        self.objectID = objectID
        self.landmarks = [landmark]

        self.counted = False
        self.up = False