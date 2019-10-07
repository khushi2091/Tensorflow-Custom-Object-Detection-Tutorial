# Training Custom Object Detector Classifier Using TensorFlow Object Detection API on Windows 10

## Summary
This repository is a tutorial for how to use TensorFlow Object Detection API to train an object detection classifier on Windows.

This readme describes every step required to get going with your own object detection classifier: 
1. [Installing Anaconda and NVIDIA GPU drivers](https://github.com/khushi2091/Tensorflow-Custom-Object-Detection-Tutorial/blob/master/README.md#1-installing-anaconda-and-nvidia-gpu-drivers)
2. [Installing TensorFlow and Preparing workspace](https://github.com/khushi2091/Tensorflow-Custom-Object-Detection-Tutorial/blob/master/README.md#2-tensorflow-installation-and-preparing-workspace)
3. [Organizing your Object Detection directory structure](https://github.com/khushi2091/Tensorflow-Custom-Object-Detection-Tutorial/blob/master/README.md#3-organizing-your-object-detection-directory-structure)
4. [Installed labelImg and annotate image datasets]()
5. [Generating training data]()
6. [Configuring training pipeline and creatinglabel map]()
7. [Train a model and monitor it's progress]()
8. [Exporting the inference graph]()
9. [Testing and using your newly trained object detection classifier]()

The repository provides all the files needed to train hand detector that can accurately detect hand. The tutorial describes how to replace these files with your own files to train a detection classifier for your own dataset. It also has Python scripts to test your classifier out on an image, video, or webcam feed.

<p align="center">
  <img src="test_data/hand.png">
</p>

## Introduction
The purpose of this tutorial is to explain how to train your own Convolutional neural network object detection classifier, starting from scratch. At the end of this tutorial, you will have a program that can identify and draw boxes around specific objects in pictures, videos, or in a webcam feed.

This tutorial also provides instructions for deploying your object classifier model in Edge devices such as Rasberry Pie, Google coral etc. In case you have linux system, You can use this repository in order to train your object detector.

The Object Detection API seems to have been developed on a Linux-based OS. We have another tutorial which has all the steps to train and deploy object detection classiifer in Linux-based OS. This tutorial uses regular TensorFlow for training the object detector clasifier but if you have GPU in your Windows machine, you can use TensorFlow-GPU instead of regular Tensorflow which will increase the training time by a factor of about 8 (3 hours to train instead of 24 hours). The GPU version of TensorFlow can also be used for this tutorial.

## Steps
### 1. Installing Anaconda and NVIDIA GPU drivers
If your system has Nvidia GPU, make sure to install the following in your system:
	Microsoft Visual Studio
	the NVIDIA CUDA Toolkit
	NVIDIA cuDNN
Follow [this YouTube video](https://www.youtube.com/watch?v=cL05xtTocmY), which shows the process for installing Anaconda, CUDA, and cuDNN. The video is made for TensorFlow-GPU v1.4, so download and install the CUDA and cuDNN versions for the latest TensorFlow version, rather than CUDA v8.0 and cuDNN v6.0 as instructed in the video. The [TensorFlow website](https://www.tensorflow.org/install/gpu) indicates which versions of CUDA and cuDNN are needed for the latest version of TensorFlow. Verify the version of CUDA drivers before installation.

I am using Windows which does not have NVIDIA GPU. Hence skipping this step in the YouTube video.

In order to install [Anaconda](https://www.anaconda.com/distribution/#download-section), Download Anaconda for Windows Python 3.7 and follow the steps mentioned [here](https://docs.anaconda.com/anaconda/install/windows/) for installing Anaconda.
#### Note: The current version of Anaconda uses Python 3.7, which is not officially supported by TensorFlow. However, when creating an Anaconda virtual environment during Step 2d of this tutorial, we will tell it to use Python 3.6)

### 2. Installing TensorFlow and Preparing workspace
Lets create an virtual environment in order to setup the Custom Tensorflow object detetction setup for your own data. From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. If Windows asks you if you would like to allow it to make changes to your computer, click Yes. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model.

#### 2.1 Set up new Anaconda virtual environment
In the command terminal that pops up, run the following command to create virtual environment:
```
C:\> conda create -n TF_object_detection pip python=3.6
```

Then, activate the environment using the following command
```
C:\> activate TF_object_detection

```

For detailed steps to install Tensorflow, follow the [Tensorflow installation instructions](https://www.tensorflow.org/install/pip#3.-install-the-tensorflow-pip-package). A typical user can install Tensorflow using one of the following commands:
```
# For CPU
(TF_object_detection) C:\>  pip install tensorflow
# For GPU
(TF_object_detection) C:\> pip install tensorflow-gpu
```

Install the other necessary packages by issuing the following commands:
```
(TF_object_detection) C:\> conda install -c anaconda protobuf
(TF_object_detection) C:\> pip install Cython
(TF_object_detection) C:\> pip install contextlib2
(TF_object_detection) C:\> pip install pillow
(TF_object_detection) C:\> pip install lmxl
(TF_object_detection) C:\> pip install jupyter
(TF_object_detection) C:\> pip install matplotlib
```

#### 2.2 TensorFlow Object Detection API setup
Download the full TensorFlow object detection repository located [here](https://github.com/tensorflow/models/archive/master.zip). Once the "models-master.zip" file will get downloaded, extract the “models-master” folder and rename “models-master” to just “models”.

#### Download pre-trained model

TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). We can take any of the pre-trained models in order to train our object detection classifier. The important thing to notice here is the performance of the model. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy.
We are planning to run the hand detector model in Edge devices such as Rasberry pie or Google coral (Note: We have separate tutorial for creating Custom Object detector classifier in Google coral where the Quantized model such as SSD-Mobilenet-V2-Quantized has been used). We have started with SSD-Inception-V2 mode but it was quite slow while running live in a video stream. I re-trained my classifier with SSD-Mobilenet-V2 model and the detection speed quite improved but with a lower accuracy. In order to improve the accuracy, the training has been done for a large no. of steps.

In this tutorial, we will use the SSD-Mobilenet-V2 model which can be downloaded [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). Open the downloaded ssd_mobilenet_v2_coco_2018_03_29.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the ssd_mobilenet_v2_coco_2018_03_29 folder to the C:\models\research\object_detection folder.
At this point, here is what your C:\models\research\object_detection folder should look like:

<p align="center">
  <img src="test_data/model.png">
</p>

Inside any of the model directory, you will find:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
    (try this out in the Jupyter notebook!)
* a config file (`pipeline.config`) which was used to generate the graph.

### 3. Organizing your Object Detection directory structure
Create a folder name as "TF_object_detection" inside C:/ drive. Move the "model" directory inside TF_object_detection such that "C:/TF_object_detection/models/research/object_detection/" this working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

        TF_object_detection
        ├─ labelImg
        │   └─ train
		│   └─ test
        ├─ models
        │   ├─ official
        │   ├─ research
        │   ├─ samples
        │   └─ tutorials
        └─ workspace
            └─ training_demo
			
#### 2.1 Protobuf compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the C:/TF_object_detection/models/research/ directory
```
(TF_object_detection) C:\> cd C:\TF_object_detection\models\research
(TF_object_detection) C:\> protoc object_detection\protos\*.proto --python_out=.
```
This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

#### 3.2 Add libraries to PYTHONPATH
When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from TF_object_detection/models/research/:
```
# From tensorflow/models/research/

(TF_object_detection) C:\> cd C:\TF_object_detection\models\research
(TF_object_detection) C:\> set PYTHONPATH=C:\TF_object_detection\models;C:\TF_object_detection\models\research;C:\TF_object_detection\models\research\slim
```
