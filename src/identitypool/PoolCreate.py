from src.detection.FaceDetection import FaceDetector
from src.recognition.FaceRecognition import FaceRecognizer

class PoolCreator:
    def __init__(self, **args):
        self.args = args
        
        self.faceDetector = FaceDetector()
        self.faceRecognizer = FaceRecognizer(self.args)
        pass
    
    