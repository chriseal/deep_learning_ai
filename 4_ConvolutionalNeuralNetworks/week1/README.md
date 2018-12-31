# Convolutional Neural Networks - Week 1

- need convolution to not have to have exponentially more nodes with larger images
  - otherwise, memory, training time, etc would be impossibly large

## Edge Detection Example

- filter:
  - contruct a matrix that looks with 1s, 0s, and -1s
  - use * to denote convolution, but in python, it means multplication
- 6x6 image * 3x3 filter = 4x4 resulting image (-1 from each side)
  - elementwise product of image and filter in each place filter will fit
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_visual.png)
- detected image seems thicker in resulting image after convolution

## More Edge Detection

- light to dark, dark to right
  - changes from all positive edge values after convolution to all negative edge values after convolution
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_vertical_and_horizontal_filters.png)
- sobel filter and schars filter or you can optimize convolution parameters with backprop (tends to work better)
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_filter_options.png)

