# Image-Autoencoder

This project implements an autoencoder network that encodes an image to its feature 
representation. The feature representation of an image can be used to conduct style 
transfer between a content image and a style image.

- The project is written in Python ```3.7``` and uses PyTorch ```1.1``` 
(also working with PyTorch ```1.3```).

- ````requirements.txt```` lists the python packages needed to run the 
project. 

### Network

The architecture consists of an pre-trained VGG-19 encoder network that was trained 
on an object recognition task. The decoder is initialized randomly and trained with
two losses.

Pre-trained models can be found in ```./models``` and loaded with the code. 

### Loss
 
The Autoencoder is trained with two losses and an optional regularizer. A 
perceptual loss measures the distance between the feature representation of the 
original image and the produced image. A per-pixel loss measures the pixel-wise 
difference between input image and output image.

### Usage

The ``configurations``-folder specifies three configurations that can be used to 
train and test the network. The project only gets the exact path to the 
configuration that is used e.g. ```python main.py './configurations/train.yaml'```.

- ``train.yaml`` trains the model from scratch. The default parameters can be found
in the file.

- ```test.yaml``` tests the model and outputs the input as well as the output image 
of the network.

- ```test_multiple.yaml``` tests several models and displays the results next to 
each other.   

### Additional Information

This project is part of a bachelor thesis which was submitted in August 2019. This 
autoencoder network makes up one chapter of the final thesis. A slightly modified 
version of the chapter can be found in this repository as a pdf-file. Also, the chapter introduces
all related formulas to this work. 

The final thesis can be found [here](https://jzenn.github.io/projects/bsc-thesis) in a corrected and modified version.