# Find Your Twin
Hello everyone, I've always wondered how casting agencies do the casting for a scene where a certain actor is young or old for a movie or TV show. I respect the art of make-up, but I am one of those who think that a different actor should play in that scene.

If we look at the developments in computer vision in recent years, there will be no need for make-up in such cases. I think that face swapping and similar approaches will make great contributions to the cinema industry in this field.

In this project, we will take a look at the problem of casting agencies, which is the first thing I wonder about. We will have an open source [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset of celebrities. We will find the face closest to the face we have given as input from this dataset.

To run the project, you need to perform 2 steps. The first is to create an identity pool, and the second is to find the identity closest to the photo given as input in this pool.

## Requirements
First of all, I suggest you to create a new environment in order not to break the environment you are using. Then you can find the required tools from `requirements.txt`
```
pip install -r requirements.txt
```
As the face recognition model, I use the PyTorch version of the ArcfaceR100 model from the [insightface](https://github.com/deepinsight/insightface) repository. You can download the weights by clicking this [link](https://onedrive.live.com/?cid=4a83b6b633b029cc&id=4A83B6B633B029CC!5577&authkey=!AFZjr283nwZHqbA) (Only backbone.pth is enough). Then place it  into `src/models/backbone.pth`.

## 1. Create Identity Pool
The identity pool to be created will process all images of a dataset one by one and save them to a pickle. If we need to go in accordance with the story, it can be said to process the images of the people in all the casting agencies one by one. This pool can be created with any dataset found on the Internet ([FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [etc.](https://analyticsindiamag.com/10-face-datasets-to-start-facial-recognition-projects/)). As I said before, I will use the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

If you want to pass this process, the pool prepared with the CelebA dataset is available at this [link](https://drive.google.com/file/d/12z5Kdk4m7ONJHC2DJcMrgF8LuPdf1BGR/view?usp=sharing).

If you are the lucky person who wants to prepare your pool in your own dataset, you should set the arguments. If your dataset is ready and you have downloaded the face recognition model, you can start creating an identity pool with the following command.

```
Format:
python create_pool.py --weightPath <Path of backbone.pth> --device <CUDA or CPU> --poolResultName <Pickle save name> --imagePaths <Your images path>

Example:
python create_pool.py --weightPath src/models/backbone.pth --device cuda:0 --poolResultName CelebrityPool2.pkl --imagePaths CelebaImages
```
## 2. Find Your Twin
You've created your pool and now it's time to try it out. First of all, you need one input image to perform the test. I left mine for testing if you want to use it :) 
There are two parameters in the command you will use here, except the ones you set when creating the pool.
```
Format:
python create_pool.py --yourImage <Input inference image> --resultImageName <Your twin image name>

Example:
python create_pool.py --yourImage cengizhan.jpg --resultImageName Twin.jpg
```
### The magic happened and you found the closest face to your own in the identity pool you created.