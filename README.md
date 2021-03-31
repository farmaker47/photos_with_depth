# Photos with depth

This is a android application that uses ML model to estimate depth in photos and produces a bokeh effect.

The model is taken from [this](https://syncedreview.com/2020/04/13/ai-transforms-rgb-d-images-into-an-impressive-3d-format/) paper and it is a CNN-based single depth estimation model. CNN-based methods have recently demonstrated promising results on estimating depth from a single image. Due to the difficulty of collecting labeled datasets, earlier approaches often focus on specific visual domains such as indoor scenes or street view. While the accuracy of these approaches is not yet competitive with multi-view stereo algorithms, that line of research and the output model is particularly promising due to the availability of larger and more diverse training datasets from relative depth annotations, multi-view stereo, 3D movies and synthetic data. For cases where only one single color image is available, they have obtained the depth estimation through a pre-trained depth estimation model. Removing the dependency on stereo or multiple images as input has made their method more widely applicable to all the existing photos.

The model that we used was written in __Pytorch__. With the provided colab notebook at the master branch you can follow along and see the conversion to ONNX, TensorFlow and finally to TensorFlow Lite to obtain the model that was finally used inside the android application. Inside the notebook you can observe all the pre and post-process of the images so an array will be available to be used with the TensorFlow Lite Interpreter. Pytorch models expect [1, 3, Width, Height] format of the inputs and so does the final TensorFlow Lite model. Due to that, usage of an array as input is mandatory and TensorFow Lite Support or TensorFow Lite Metadata libraries were not used in this project.

The output of the model is an array of [1, 1, Width, Height] shape. This array is converted to a grayscale image and then on screen you can observe the input image and a grayscale one with the depth estimation in various tones of gray. Selecting specific values of pixels above a certain number we focus on the objects inside the image that are closer to the camera. That objects remain unchanged and the background is converted to B/W, blurred or sepia. Below you can see some mobile selfie screenshots:

<img src="images/george_1.jpg" width="280" height="404"> <img src="images/george_2.jpg" width="280" height="404"> <img src="images/george_3.jpg" width="280" height="404">

and screenshots when the background camera is used inside a room:

<img src="images/nornal_1.jpg" width="280" height="404"> <img src="images/normal_2.jpg" width="280" height="404"> <img src="images/normal_3.jpg" width="280" height="404">
