import argparse

from src.recognition.FaceRecognition import FaceRecognizer


class CelebrityFinder:
    def __init__(self, **args):
        self.recognizer = FaceRecognizer(**args)
        #TODO add face recognition model
        #TODO add identity pool
        #TODO add cosine similarity
        pass

    def main(self):
        pass

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--weightPath", default="src/models/backbone.pth", help="Face recognition weight path", type=str)
    args.add_argument("--device", default="cuda", help="cpu, cuda (If have multiple GPU may use like cuda:0, cuda:1 etc. )", type=str)

    arguments = vars(args.parse_args())

    
    manager = CelebrityFinder(**arguments)
    manager.main()
