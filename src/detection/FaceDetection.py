from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.faceDetectionModel = MTCNN()
        print("Face detector model initialized!")

    def main(self):
        pass