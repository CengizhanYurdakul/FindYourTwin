import numpy as np

def create_arcface_template()->np.array:
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
