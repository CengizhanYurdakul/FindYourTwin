import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity 

from src.detection.FaceDetection import FaceDetector
from src.recognition.FaceRecognition import FaceRecognizer

class CelebrityFinder:
    def __init__(self, **args):
        self.args = args
        #TODO add feature to choose update pool or create new one
        
        self.faceDetector = FaceDetector(**self.args)        
        self.faceRecognizer = FaceRecognizer(**self.args)
        
        with open(self.args["poolResultName"], "rb") as f:
            self.identityPool = pickle.load(f)
        print("Identity pickle loaded!")
        
        self.maxSimilarityImageName = None
        self.maxSimilarity = -1
        
    def visualizeCelebrityVersion(self):
        celebrityImage = cv2.imread(os.path.join(self.args["imagePaths"], self.maxSimilarityImageName))
                
        cv2.imwrite(self.args["resultImageName"], celebrityImage)
        print("Your similarity with your celebrity: %s" % round(self.maxSimilarity, 3))
 
    def find(self):
        inputImage = cv2.imread(self.args["yourImage"])
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        
        alignedImage = self.faceDetector.alignFace(inputImage)
        
        if alignedImage is None:
            raise Exception("No face detected!")
        
        inputIdentity = self.faceRecognizer.extractIdentity(alignedImage)
        
        for celebrityImageName in tqdm(self.identityPool.keys()):
            celebrityIdentity = self.identityPool[celebrityImageName]
            cosineSimilarity = float(cosine_similarity(inputIdentity.cpu(), celebrityIdentity))
            if cosineSimilarity > self.maxSimilarity:
                self.maxSimilarity = cosineSimilarity
                self.maxSimilarityImageName = celebrityImageName
        
        self.visualizeCelebrityVersion()