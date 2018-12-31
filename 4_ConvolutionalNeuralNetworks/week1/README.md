# Convolutional Neural Networks - Week 1

- need convolution to not have to have exponentially more nodes with larger images
  - otherwise, memory, training time, etc would be impossibly large

## Edge Detection Example

- filter:
  - contruct a matrix that looks with 1s, 0s, and -1s
  - use * to denote convolution, but in python, it means multplication
- 6x6 image * 3x3 filter = 4x4 resulting image 
- nxn image * fxf filter = (n-f+1)x(n-f+1) resulting image 
  - elementwise product of image and filter in each place filter will fit
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_visual.png)
- detected image seems thicker in resulting image after convolution

## More Edge Detection

- light to dark, dark to right
  - changes from all positive edge values after convolution to all negative edge values after convolution
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_vertical_and_horizontal_filters.png)
- sobel filter and schars filter or you can optimize convolution parameters with backprop (tends to work better)
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_filter_options.png)

## Padding

- nxn image * fxf filter = (n-f+1)x(n-f+1) resulting image 
- every time you apply a convolution is your image shrinks
- throws away info in edged of picture: pixels on corners are used less often in convolutions than pixels in the middel
--> pad image with border of 1 pixel on each side with zeros
- with padding
  - add 1 px of 0's around entire image, so after convolution, resulting image has same size as original image
  - add 2+ px to border if you want
  - resulting image is (n+2p-f+1)x(n+2p-f+1), where p is number of pixels added to border
- valid convolutions (bad name)
  - "valid": nxn * fxf --> (n-f+1)x(n-f+1)
  - "same": , p=(f-1)/2, most common
    - filter is 5, border/p should be = 2.... p=(5-1)/2
    - f is almost always odd  
      - 3x3, 5x5, 7x7 are most common conventions
      - Ng almost always just uses odd dimensions in f
      
## Strided Convolutions

- stride = s = 2
- move filter over by `stride` number of pixels at a time, 
- resulting image size: ((n+2p-f)/s + 1) x ((n+2p-f)/s + 1)
- but if the dimensions aren't an integer, round down / take the floor
  - floor((n+2p-f)/s + 1) x floor((n+2p-f)/s + 1)
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_resulting_size.png)
- cross correlation vs. convolution
  - math textbook will call cross-correlation to what we're calling convolution now
  - technically, convolution in math textbook involves flipping the filter diagonally
  - but we aren't doing this flipping, and it isn't done in deep learning - bc it's not necessary

## Convolutions Over Volume

- Convolutions in RGB image
  - 6x6x3 convolved with 3x3x3 filter
  - (height x width x #channels) * (filter_height x filter_width x #channels)
    - sometimes, literature calls channels depth, but we're using channels
  - can detect edges in green, red, or blue channels independently
  - can have RGB all be same if you don't care about which channel has an edge
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_on_3d_volumes.png)
- this image assumes no padding and stride of 1:
- very powerful
  - can now detect many edges in many channels and output has same number of channels that you are detecting
- ![img](https://github.com/chriseal/deep_learning_ai/4_ConvolutionalNeuralNetworks/blob/master/week1/4wk1_convolution_multiple_filters.png)

## One Layer of a Convolutional Network



