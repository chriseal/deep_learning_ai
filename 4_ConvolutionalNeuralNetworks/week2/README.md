# Convolutional Neural Networks - Week 2

- use case studies, because a similar architecture will likely transfer to related problems ish

## Classic Networks

- LeNet - 5 : predict which hand written digit
  - average pooling no longer used (use max instead)
  - people didn't usually apply SAME pooling back then, so dimensions got reduced with each step
  - 60K parameters; small by today's standard
    - today, you see NN's with 10-100M parameters
  - H and W tend to get smaller, number of channels tend to get larger as you go deeper into the NN
  - conv-pool-conv-pool-FC-FC-output still common today



- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week2/deep_nn_notation.png)
