# Convolutional Neural Networks - Week 2

- use case studies, because a similar architecture will likely transfer to related problems ish

## Classic Networks

- LeNet - 5 : predict which hand written digit
  - 32x32x1
  - average pooling no longer used (use max instead)
  - people didn't usually apply SAME pooling back then, so dimensions got reduced with each step
  - 60K parameters; small by today's standard
    - today, you see NN's with 10-100M parameters
  - H and W tend to get smaller, number of channels tend to get larger as you go deeper into the NN
  - conv-pool-conv-pool-FC-FC-output still common today
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_LeNet.png)
- AlexNet, 2012
  - 227x227x3
  - similar to LeNet, but much bigger ~60M parameters
  - used ReLU
  - Local Response Normalization (LRN) - not used much, normalizes params
  - easier paper to read
  - output softmax of 1000 categories
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_AlexNet.png)
- VGG-16
  - simpler network, less params
  - pretty large ~138M parameters
  - VGG-19 does slighly better than VGG-16 (most ppl just use VGG-16)
  - second easiest paper to read
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_VGG16.png)

## ResNets

- activations can skip a layer and re-enter in a deeper layer
- allows for deeper NN's
- really helps with expanding and vanishing gradient values
- residual block
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_residual_block.png)
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_resNet.png)

## Why ResNets Work

- in practice, normal NN's get worse on training set at a certain point after adding too many layers
- ResNets don't suffer from this
- makes it easier to learn identity function, and anything on top of that is icing on the cake
- so it can't hurt performance
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_resNet_why_work.png)
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_resNet_example.png)

## Networks in Networks and 1x1 Convolutions

- it's as if you're multiplying at one HxW position across all the channels to get a scalar value and then you take the ReLU of that
- network in a network
- Lin et al 2013
- can carry out very complex functions
- can be used to reduce the number of channels
- pooling filters shrinks n_h and h_w
- 1x1 convolution shrinks n_c
- adds nonlinearity
- (1 x 1 x n_channels) in previous layer, and apply as many filters as you want
- allows your network to learn more complicated functions, due to introduction of non-linearity 
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_1x1_conv.png)

## Inception Network Motivation

- inception layer
  - stack different convolution sizes
  - do all pooling sizes
  - Szedy et al 2014
- comes at a computational cost
  - using a 1x1 convolution as an intermediate step can reduce computation cost by a factor of 10
    - aka, a 'bottleneck' layer
  - adding a 1x1xn_c intermediate convolution layer doesn't seem to hurt performance
  
## Inception Network

- one inception module:
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_inception_module.png)
- adds a softmax layer at intermediate layers to ensure the features are decent at predicting the target variable
  - this adds a bit of a regularization effect which reduces overfitting
(https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/4wk2_inception_network.png)
- has many variations... combining this with resnet can sometimes produce better results

## Using Open-Source Implementation

## Transfer Learning

- can download open source weights from someone who's spent weeks or months and many GPU's to come up with
- can speed up progress
- if you have more data, the number of layers you freeze could be smaller and the number of layers you train could be larger
- can also just use the weights as initialization (if you have a lot of data) 
  - with different softmax output of course
  
## Data Augmentation

- almost all computer vision tasks, having more data will help
- very common technique in computer vision
- true whether you're using transfer learning or training from scratch
- flip vertically 
- random cropping - not a perfect method, but in practice it works as long as subset is big enough representation of the image itself
- rotation, shearing, local warping <- acceptable but used less
- color shifting - add +20R, -20G, +20B for example... in reality, you'd draw from random samples
  - PCA color augmentation - adjusts color relative to existing ratios
- can implement distortions in real time during training stage
- has its own set of hyperparameters
  - use someone else's open-source solution to start out
  











