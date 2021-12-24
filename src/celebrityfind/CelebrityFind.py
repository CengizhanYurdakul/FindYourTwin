import argparse

from src.detection.FaceDetection import FaceDetector
from src.recognition.FaceRecognition import FaceRecognizer



class CelebrityFinder:
    def __init__(self, **args):
        self.recognizer = FaceRecognizer(**args)
        self.detector =  FaceDetector()
        #TODO add identity pool
        #TODO add cosine similarity
        #TODO add face alignment

        pass

    def main(self):
        pass

