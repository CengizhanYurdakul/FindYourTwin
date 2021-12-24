import cv2
from facenet_pytorch import MTCNN
from skimage import transform as trans

from src.utils.utils import *
class FaceDetector:
    def __init__(self):
        self.faceDetectionModel = MTCNN()
        print("Face detector model initialized!")
        
        self.alignmentTemplate = alignmentTemplate()
        self.similarityTransform = trans.SimilarityTransform()
        
    def alignFace(self, inputImage:np.array)->np.array:
        """
        It finds the largest face in the photo given as input and performs alignment in accordance with the template.

        Args:
            inputImage (np.array): The image that needs to be aligned.

        Returns:
            np.array: Aligned face.
        """
        faceInformations = self.detectFace(inputImage)
        
        boundingBox, landmark5 = parseDetectionResult(faceInformations)
        
        self.similarityTransform.estimate(landmark5, self.alignmentTemplate)
        transformMatrix = self.similarityTransform.params[0:2, :]
        
        alignedImage = cv2.warpAffine(inputImage, transformMatrix, (112, 112)).astype(np.uint8)

        return alignedImage

    def detectFace(self, inputImage:np.array)->tuple:
        """
        Finds the faces in the given photograph and the landmarks of these faces.

        Args:
            inputImage (np.array): The image to be run face detection.

        Returns:
            tuple: A tuple containing the coordinates of the detected faces, confidence and 5 landmarks, respectively.
        """
        faceInformations = self.faceDetectionModel.detect(inputImage, landmarks=True)
        return faceInformations