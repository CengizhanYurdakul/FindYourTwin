import os
import cv2
import pickle
from tqdm import tqdm

from src.detection.FaceDetection import FaceDetector
from src.recognition.FaceRecognition import FaceRecognizer

class PoolCreator:
    def __init__(self, **args):
        self.args = args
        
        self.faceDetector = FaceDetector(**self.args)        
        self.faceRecognizer = FaceRecognizer(**self.args)
        
        self.imagePaths = os.listdir(self.args["imagePaths"])
        
        self.identityDictionary = {}
        
        self.undetectedImages = 0
        
    def updateIdentityDictionary(self, imageName, outputIdentity):
        self.identityDictionary[imageName] = outputIdentity
        
    def savePickle(self):
        with open(self.args["poolResultName"], "wb") as f:
            pickle.dump(self.identityDictionary, f, pickle.HIGHEST_PROTOCOL)
    
    def create(self):
        for imageName in tqdm(self.imagePaths):
            inputImage = cv2.imread(os.path.join(self.args["imagePaths"], imageName))
            inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
            
            alignedImage = self.faceDetector.alignFace(inputImage)
            
            if alignedImage is None:
                self.undetectedImages += 1
                continue
            
            outputIdentity = self.faceRecognizer.extractIdentity(alignedImage)
            
            self.updateIdentityDictionary(imageName, outputIdentity)
            
        
        self.savePickle()
        print("%s image could not be detected!" % self.undetectedImages)
        
    