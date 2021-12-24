import torch
import numpy as np

from src.recognition.Backbone import IResNet

class FaceRecognizer:
    def __init__(self, **args):
        self.args = args

        self.faceRecognitionModel = IResNet()
        self.faceRecognitionModel.load_state_dict(torch.load(self.args["weightPath"]))
        self.faceRecognitionModel.eval()
        self.faceRecognitionModel.to(self.args["device"])

        print("Face recognition model weights initialized!")
        
    def preprocess(self, inputImage:np.array)->torch.Tensor:
        """
        It preprocesses the array, which will enter the face recognition model, such as permute and normalization.

        Args:
            inputImage (np.array): The image to be preprocessed.

        Returns:
            torch.Tensor: A preprocessed tensor.
        """
        inputImage = np.transpose(inputImage, (2, 0, 1))
        inputImage = torch.from_numpy(inputImage).unsqueeze(0).float()
        inputImage.div_(255).sub_(0.5).div_(0.5)
        return inputImage.to(self.args["device"])

    def extractIdentity(self, inputImage:np.array)->torch.Tensor:
        """
        It gives the input to the face recognition model and takes the identity output.

        Args:
            inputImage (np.array): The image whose identity is desired to be obtained.

        Returns:
            torch.Tensor: Identity
        """
        inputImage = self.preprocess(inputImage)
        inputIdentity = self.faceRecognitionModel(inputImage)
        return inputIdentity