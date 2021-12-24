import argparse

from src.celebrityfind.CelebrityFind import CelebrityFinder

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--weightPath", default="src/models/backbone.pth", help="Face recognition weight path", type=str)
    args.add_argument("--device", default="cuda", help="cpu, cuda (If have multiple GPU may use like cuda:0, cuda:1 etc. )", type=str)

    arguments = vars(args.parse_args())

    
    manager = CelebrityFinder(**arguments)
    manager.main()