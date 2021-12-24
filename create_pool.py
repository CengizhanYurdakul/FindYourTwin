import argparse

from src.identitypool.PoolCreate import PoolCreator

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    args.add_argument("--weightPath", default="src/models/backbone.pth", help="Face recognition weight path", type=str)
    args.add_argument("--device", default="cuda:0", help="cpu, cuda (If have multiple GPU may use like cuda:0, cuda:1 etc.)", type=str)
    args.add_argument("--poolResultName", default="CelebrityPool.pkl", help="Output pickle which is including all identities", type=str)
    args.add_argument("--imagePaths", default="CelebaHQImages", help="Images path that includes all images", type=str)

    arguments = vars(args.parse_args())

    
    manager = PoolCreator(**arguments)
    manager.create()