import torch

from src.recognition.backbone import IResNet

class FaceRecognizer:
    def __init__(self, **args):
        self.args = args

        self.faceRecognitionModel = IResNet()
        self.faceRecognitionModel.load_state_dict(torch.load(self.args["weightPath"]))
        self.faceRecognitionModel.eval()
        self.faceRecognitionModel.to(self.args["device"])

        print("Face recognition model weights initialized!")

    def extractIdentity(self, inputImage):
        pass

    def main(self):
        pass