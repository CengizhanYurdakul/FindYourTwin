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
    boundingBoxes = allInformations[0]
    landmark5 = allInformations[2]
    index = findBiggestFace(boundingBoxes)
    return boundingBoxes[index], landmark5[index]

def findBiggestFace(boundingBoxes:np.array)->int:
    """
    It takes the largest face among the detected images of more than one face.

    Args:
        boundingBoxes (np.array): Coordinates of detected faces.

    Returns:
        int: The index with the largest face.
    """
    biggestIndex = None
    biggestArea = 0
    for c, i in enumerate(boundingBoxes):
        boundingBox = parseBoundingBox(i)
        area = boundingBox[2] * boundingBox[3]
        if area > biggestArea:
            biggestArea = area
            biggestIndex = c
    return biggestIndex
    
def parseBoundingBox(boundingBox:np.array, type:str="xywh")->np.array:
    """
    Converting the coordinates of the points where the faces are located to the formats "xyxy" and "xywh" to use different types.

    Args:
        boundingBox (np.array): Coordinates of detected faces.
        type (str, optional): In the coordinates of the position of the face, "xyxy" gives the x,y coordinates of the upper left 
        point and the x,y coordinates of the lower right point, and "xywh" gives the x,y coordinates of the upper left point, its 
        height and width.. Defaults to "xywh".

    Raises:
        Exception: If a format other than these two formats is entered.

    Returns:
        np.array: An array of bounding boxes in "xywh" or "xyxy" format.
    """
    if type == "xyxy":
        parsedBoundingBox = [boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]]
    elif type == "xywh":
        parsedBoundingBox = [boundingBox[0], boundingBox[1], boundingBox[2] - boundingBox[0], boundingBox[3] - boundingBox[1]]
    else:
        raise Exception("Unknown type of parse bounding box!")
    
    return np.array(parsedBoundingBox)