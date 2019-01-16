# Vangogh Crazy World Android Applications

## Neural style transfer

Neural style transfer is the process of taking the style of one image then applying it to the content of another image.

Offering you a variety of beautiful styles some of which are paintings by famous artists like Starry Night by Van Gogh.


## Vincent van Gogh
For more Vincent van Gogh, please refer to [wiki](https://en.wikipedia.org/wiki/Vincent_van_Gogh)

## Tensorflow Lite
The model in this project is written in Tensorflow Lite

For more Tensorflow Lite, please refer to [Tensorflow Lite](https://www.tensorflow.org/lite/)


## Experience

We would like to bring you to Vangogh Crazy World, insight the sample code, you can use pre-trained model or you can try to train your own models.

After that you can decided to deploy on various devices which running on to Windows or Android and even OS independent web pages.

## Table of Contents (Optional)

* Features
* Getting Started
* Prerequisites
* ...

## Features

In this project, it will provide the following packages
* Training Van Gogh gallery with Python
* Inference with real time camera and still images and 
  * Deployment on Windows applications
  * Deployment on Android applications
  * Deployment on web pages


## Getting Started


### Getting the Code

```
git clone https://github.com/acerwebai/VangoghCrazyWorld.git
```

### Run pre-built apk
You can download the pre-built apk from here


## Converting Model

While in the training process, we build up the model and freeze it.

If you would like to kow how to train the neural style transfer models, please refer to [VangoghCrazyWorld](https://github.com/acerwebai/VangoghCrazyWorld)

Now we can used the frozen model and deploy various devices in Windows or Android, or even OS independent web pages.

### Tensorflow Lite

We can convert Tensorflow frozen.pb to Tensorflow Lite format via [TensorFlow Lite Converter](https://www.tensorflow.org/lite/convert/)

Once you have ever installed the tensorflow during the training process, you can use the following to convert the Tensorflow model as Tensorflow Lite format

```
tflite_convert \
  --graph_def_file=frozen.pb \
  --output_file=graph.lite \
  --output_format=TFLITE \
  --input_shape=1,256,256,3 \
  --input_array=X_content \
  --output_array=add_37 \
  --inference_type=FLOAT 
```

where
* --graph_def_file: is the Tensorflow frozen graph file
* --output_file: is the output Tensorflow Lite format file
* --output_format: TFLITE
* --input_shape: the shape of input
* --input_array: the input array name, in our case, should be X_content
* --output_array: the output array name, in our case, should be add_37
* --inference_type: FLOAT or QUANTIZED_UINT8

Tensorflow Lite is easy to be deployed in Android application and can get benifit of the accerleratio of [Android Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/)

## Android Applications

Now we start to build the Tensoflow Lite model into Android application

### Android Studio

Install Android Studio from [Android web site](https://developer.android.com/studio/)

### Launch Project

Open the project gradle files and see all project files

### Build Project

Build project

### Make APK for Deployment

Make release apk for installing on Android devices

### Deploy

You can deploy the apk on any Android devices

* Android phone
* Chromebook


### Deploy

Make your own Android applications



## Implementation Details

The implementation is based on the [Fast Style Transfer in TensorFlow from ](https://github.com/lengstrom/fast-style-transfer) from [lengstrom](https://github.com/lengstrom/fast-style-transfer/commits?author=lengstrom)

It use roughly the same transformation network as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output tanh layer is slightly different. We use a loss function close to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation (e.g. we use relu1_1 rather than relu1_2). Empirically, this results in larger scale style features in transformations.

### Paper

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/eccv16/)


### Framework

[Tensorflow Lite](https://www.tensorflow.org/lite/)

### Model

VGG16

### Optimization



## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## Versioning

Version and release note

## Team

Team member

## FAQ

Frequently ask questions

## Support

Who you should contact with, email


## License

This project is licensed under the Apache License 2.0, see the [LICENSE.md](LICENSE)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* ...


