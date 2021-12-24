import numpy as np

def alignmentTemplate()->np.array:
    """
    The alignment template of the face to be given as input to the face recognition model.

    Returns:
        np.array: X and y axis values of eye, mouth and nose points.
    """
    return np.array([[38.2946, 51.6963],
                     [73.5318, 51.5014],
                     [56.0252, 71.7366],
                     [41.5493, 92.3655],
                     [70.7299, 92.2041]], dtype=np.float32)

def parseDetectionResult(allInformations:np.array)->np.array:
    """
    The method that splits the detection outputs and takes the ones to be used.

    Args:
        allInformations (np.array): An array with bounding box, confidence and 5 landmarks, respectively.

    Returns:
        np.array: Bounding box and 5 landmark information.
    """
    return allInformations[0], allInformations[2]