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

  
