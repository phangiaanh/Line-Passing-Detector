class ProcessingFrame:
    def __init__(self, id, image):
        self.id = id
        self.image = image
        self.isSomeone = False
        self.box = []
        self.color = []
        self.processingTime = {}
        self.total = []

