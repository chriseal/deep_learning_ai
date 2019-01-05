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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_localization_ex.png)
- In reality, you can use logistic loss for p_c, mean squared error for bounding box, and categorical cross entropy for multiclass classification
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_loss_function_multiclass.png)

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

## Object Detection

- sliding windows
- closely cropped images
- train convNet on closely cropped images
  - classify closely cropped images as 1, 0 whether or not the cropped image contains a car
  - go through every region of every position of a larger image and train a convNet on this
  - after going through entire image, use a bigger crop size and make associated predictions with ConvNet
  - repeat one or more times
  - if you use a big/coarse stride, it reduces number of predictions but may also reduce performance
  - too small stride comes at computational cost
- before the rise of NN's, people used linear classifier over hand-engineered features
  - sliding window works okay
- but with NN's the cost of a sliding window is too high 
  --> sliding window's object detector can be implemented convolutionally

## Convolutional Implementation of Sliding Windows

- turning fully connected (FC) layer into convolutional layers 
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_turning_FC_into_convolution.png)
- say, your test image has bigger area than test examples --> use a sliding window to classify
  - train on 14x14x3 and test on 16x16x3 --> 4 different predictions
  - lots of repeated calculations
  --> use a convolution to share calculations when making predictions
  - convolutional implementation at test time reduces calculations by a lot
  - run through same calculations as trained on (different image sizes result)
  - all four predictions are taken into account in final step of convoluted forward prediction pass
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_convolutional_implementation_of_sliding_windows.png)
- only requires one forward pass to make prediction of all crops in an image

## Bounding Box Predictions

- problem: bounding box positions may not line up exactly with a car
--> YOLO algorithm "You Only Look Once"
- make 9 grid cells in picture
- apply sliding window in each grid cell to the labels
- 9 label arrays for full image (1 label array for each grid cell)
- labels assign object to grid cell based on midpoint of object (dots in picture)
- target output labels will be 3x3x8, because you have 8 values in each y mini array and you have 3x3 grid cells
- to train NN:
  - input is 100x100x3, as usual
  - CONV --> MAXPOOL --> .. --> 3x3x8
- advantage is algorithm outputs precise bounding boxes
- might use a 19x19x8 grid cell (finer grain) in reality, so less of a chance that multiple objects are in same grid cell
- boundary box is much better, outputs boundary box explicitly
  - any aspect ratio of bounding box, boundary box not affected by stride size of sliding swindow's classifier
  - lots of shared calculations (much better than classifying independently on each crop)
- trains on entire image so can be used in predictions on entire image
- runs very fast, so even works for real-time object detection
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_yolo_algorithm.png)
- specifying the boundary boxes
  - midpoint is defined relative to grid cell from 0,0 to 1,1 x,y space 
    - x and y must be bw 0 and 1
  - height and width are specified as proportional to size of grid cell
    - H and W can be >1 if object spans multiple grid cells
  - there are slightly better ways to label these examples, using YOLO paper recommendations, but they are much more complicated
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_boundary_box_labels.png)

## Intersection Over Union (IoU)

- IoU computes the overlapping area bw truth and predicted boundary boxes
- size of overlap / size of predicted area
- "correct if IoU >= 0.5 (0.5 is just a convention... you could use 0.6, etc... Ng rarely sees people drop it below 9.5)
- measures how similar two boxes are to each other

## Non-max Suppression

- makes sure your algorithm detects each unique object only once
- multiple grid cells may think they've found the car
- running image classification and localization algorithm on every grid cell
- possible that many grid cells have p_c large (probability of an object)
- non-max suppression cleans up these multi-detections
- first takes largest p_c and then looks at nearby high p_c's with high amount of overlap with first area
  - gets rid of any other positive predictions with a high IoU with the highest p_c prediction
- 19x19 grid cell --> 19x19x8 target/y variable for each image
  - discard all boxes with p_c <= 0.6 (or 0.5 or whatever)
  - discard any remaining boxes with IoU >= 0.5 with currently chosen box
  - does this for each class 
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_nonMaxSuppresion.png)

## Anchor Boxes

- What if you want to predict multiple objects in one grid cell?
  - Anchor boxes help with this
- overlapping objects
  - predefine two different shapes/anchor box
    - anchor box 1, anchor box 2, ... anchor box number of objects you want to predict in each grid
  - use anchor box most similar to object type to relate one anchor box to one category of classification for each given image independently
    - e.g. use horizontal anchor box for a car and vertical anchor box for a human for a given image
- pick which anchor box has a higher IoU with actual object
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_anchor_box_algorithm.png)
- this algorithm doesn't handle more objects in a grid cell than coded into algorithm (use a tiebreaker code)
  - doesn't happen much
- this algorithm also doesn't handle two objects with same shape in grid cell (use a tie breaker)
- pro: allows learning algorithm to specialize in predicting certain kinds of shapes
  - some outputs are great at predicting vertical shapes, some great at predicting horizontal shapes, etc
- can use a K-Means algorithm to choose which anchor boxes you use, based off of labeled data
  - shapes that more closely match your training set

## YOLO Algorithm

- 












