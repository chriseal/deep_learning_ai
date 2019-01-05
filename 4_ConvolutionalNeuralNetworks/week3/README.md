# Convolutional Neural Networks - Week 3 - Localization

## Object Localization

- classification
- classification with localization
  - multiple classification with localization
  - pedestrian, car, motorcycle, background
- to localize a car, output a bounding box (4 numbers)
  - add 4 numbers to each class label probability in softmax
  - b_x, b_y, b_w, b_h
  - output: is there an object? bounding box, class probabilities
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk3_localization_ex.png)
- In reality, you can use logistic loss for p_c, mean squared error for bounding box, and categorical cross entropy for multiclass classification
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk3_loss_function_multiclass.png)

## Landmark Detection

- where is the corner of someone's eye?
  - x and y coordinate
- what about multiple points along an eye?
  - x, y coordinate for each landmark
  - plus a binary of whether or not the image exists
- can output pose of a person based on bone landmarks
- seems like a simple idea, but very powerful
- need consistent labels
- key building block for special graphics effects, emotion detection




