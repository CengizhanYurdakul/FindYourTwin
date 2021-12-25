# Find Your Famous Version
Hello everyone, I've always wondered how casting agencies do the casting for a scene where a certain actor is young or old for a movie or TV show. I respect the art of make-up, but I am one of those who think that a different actor should play in that scene.

If we look at the developments in computer vision in recent years, there will be no need for make-up in such cases. I think that face swapping and similar approaches will make great contributions to the cinema industry in this field.

In this project, we will take a look at the problem of casting agencies, which is the first thing I wonder about. We will have an open source celeba dataset of celebrities. We will find the face closest to the face we have given as input from this dataset.

To run the project, you need to perform 2 steps. The first is to create an identity pool, and the second is to find the identity closest to the photo given as input in this pool.

## Requirements
First of all, I suggest you to create a new environment in order not to break the environment you are using. Then you can find the required tools from `requirements.txt`
```
pip install -r requirements.txt
```

## Create Celebrity Identity Pool
