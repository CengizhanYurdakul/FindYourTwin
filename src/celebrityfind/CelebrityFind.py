import cv2
import argparse

from src.detection.FaceDetection import FaceDetector
from src.recognition.FaceRecognition import FaceRecognizer

class CelebrityFinder:
    def __init__(self, **args):
        self.recognizer = FaceRecognizer(**args)
        self.detector =  FaceDetector()
        #TODO add identity pool
        #TODO add cosine similarity
        pass

    def main(self):
        img = cv2.cvtColor(cv2.imread("test1.jpg"), cv2.COLOR_BGR2RGB)
        aligned = self.detector.alignFace(img)
        identity = self.recognizer.extractIdentity(aligned)
        pass

