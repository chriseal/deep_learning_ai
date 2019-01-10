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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_visual.png)
- detected image seems thicker in resulting image after convolution

## More Edge Detection

- light to dark, dark to right
  - changes from all positive edge values after convolution to all negative edge values after convolution
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_vertical_and_horizontal_filters.png)
- sobel filter and schars filter or you can optimize convolution parameters with backprop (tends to work better)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_filter_options.png)

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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_resulting_size.png)
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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_on_3d_volumes.png)
- this image assumes no padding and stride of 1:
- very powerful
  - can now detect many edges in many channels and output has same number of channels that you are detecting
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_multiple_filters.png)

## One Layer of a Convolutional Network

- add in bias term to each convolution (real number)
  - apply elementwise to resulting image
  - apply relu to resulting matrix
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_layer.png)
- convolutions can guard against overfitting, because no matter how many X feature you have, after applying convolutions, your resulting layer will always have same number of parameters to learn
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_layer_num_params.png)
- height and width (and all channels) have same formula for shape
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_summary_of_notation.png)

## Simple Convolutional Network Example, ConvNet

- structure of a convolutional NN, example
  - 37 comes from (n+2P-f)/s+1
- a typical structure of convolutional NN
- as you go deeper in NN, dimensions of images will gradually scale down
- flatten last layer and input into logistic regression
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_convolution_typical_structure.png)
- most NN's have a few pooling and a few fully connected layers in addition to convolutional layer

## Pooling Layers

- max pooling
  - each of output of filter is biggest number in eaach region
- hyperparameters are f/filter and s/stride in pooling layer
  - f = 2, s = 2 are common choices
    - has the affect of reducing H and W by half
  - f=3, s=2 sometimes used
- if this feature is detected anywhere in the filtering layer, it's still accounted for
- but it works well, and regardless of theoretical discussion
- no parameters, but does have f and s as hyperparameters
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_max_pooling_explanation.png)
- max pooling computation is calculated independently on each channel
- average pooling used rarely
  - very deep NN may use this for data compression
- no padding used in max pooling, p = 0 by far most common choice
- floor [ (n_h - f)+ 1 ] x floor [ (n_x - f)+ 1 ]

## CNN Example

- inspired by LeNet-5
- different conventions about number of layers when using max-pooling
  - but max pooling has no parameters, so we're going to just combine conv-1 and pool-1 as one layer-1
- reasonably typical structure
  - Conv+MaxPool --> Conv2+MaxPool2 --> (flatten) --> FC3 --> FC4 --> softmax
- use architecture that has worked for someone else
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_conv_typical_structure.png)
- number of HxW dimensions gets smaller, number of channels gets larger as you go deeper into NN
- Activation size tends to gradually get smaller the deeper you go into NN, common trend
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week1/4wk1_conv_typical_structure_param_chart.png)

## Why Convolutions?

- number of connection in FC layer with large image vector input would be unfeasibly large
- reduces complexity of model
- **parameter sharing**: if vertical edge detection is useful in one part of image, it will be helpful in another part of an image
  - moves features all around the image
- **sparsity of connections**: each layer depends only on a small number of inputs (because of filtering , 33 filter only takes into account 9 input features)
- conv: can be trained with smaller datasets
- conv: less prone to overfitting
- translation invariance: cat moving from one location in image to another
  - conv' account for this because you're applying the same feature all around the image
- same cost function as previous models (gradient descent, momentum, rms prop, adam)




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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_LeNet.png)

- AlexNet, 2012
  - 227x227x3
  - similar to LeNet, but much bigger ~60M parameters
  - used ReLU
  - Local Response Normalization (LRN) - not used much, normalizes params
  - easier paper to read
  - output softmax of 1000 categories
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_AlexNet.png)
- VGG-16
  - simpler network, less params
  - pretty large ~138M parameters
  - VGG-19 does slighly better than VGG-16 (most ppl just use VGG-16)
  - second easiest paper to read
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_VGG16.png)

## ResNets

- activations can skip a layer and re-enter in a deeper layer
- allows for deeper NN's
- really helps with expanding and vanishing gradient values
- residual block
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_residual_block.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_resNet.png)

## Why ResNets Work

- in practice, normal NN's get worse on training set at a certain point after adding too many layers
- ResNets don't suffer from this
- makes it easier to learn identity function, and anything on top of that is icing on the cake
- so it can't hurt performance
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_resNet_why_work.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_resNet_example.png)

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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_1x1_conv.png)

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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_inception_module.png)
- adds a softmax layer at intermediate layers to ensure the features are decent at predicting the target variable
  - this adds a bit of a regularization effect which reduces overfitting
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_inception_network.png)
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

## State of Computer Vision

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week2/4wk2_data_vs_transfer_learning.png)
- ensembling almost never used in production because of added cost of predction; taking the average of 3-15 indepentently trained models
- multi-crop at test time
  - take multiple crops of an image and average the results
  - apply data augmentation to test set
- start out with someone else's architecture, open source implementations, pretrained models


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

- output shape is:
  - height of image, width of image, (# of anchors, 5+# of choices (object types you're classifying))
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_yolo_making_predictions.png)
- numbers will be output for null y's, but you can ignore those
- independently run non-max suppression for each class
- YOLO is one of the best image classification algorithms, combining the best ideas across all of computer vision
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_yolo_making_preds2.png)

# Region proposal - R-CNN (optional)

- Ng doesn't use this that much, but it still has an important body of work in image classification
- runs a segmentation algorithm to decide what could be objects
- can reduce training time, especially if you're using anchor boxes with different shapes and/or at multiple scales
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week3/4wk3_region_proposal_explanation.png)
- R-CNN algorithm is still quite slow
  - classify proposed regions one at a time
  - output label + bounding box that is independent of segmentation algorithm
- Fast R-CNN
  - R-CNN with convolutional implementation to classify all regions at once
  - proposing regions segmentation is still slow
- Faster R-CNN: use convolutional NN to propose regions
- but all of these are still slower than YOLO
- Ng thinks that region proposals are interesting, but likes 1-step YOLO over 2-step segmentation + classification
  - better long term solution he thinks


# Convolutional Neural Networks - Week 4

## What is face recognition?

- face verification
  - 1 to 1 problem
  - given image and name/ID output whether or not that image is that person
- face recognition
  - much harder
  - have to compare to entire database --> much more likely to have errors
live face detection (is a picture or a live human)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_verification_vs_recognition.png)

## One Shot Learning

- only have one image in your training set
- learn a similarity function
  - degree of difference bw two images
  - small number if same person, large number if not
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_one_shot_learning.png)

## Siamese Network

- two identical CNN weights applied to two images 
- compare last FC layer arrays for normed diff
- see DeepFace paper
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_siamese_learning.png)
- Goal: want encoding that makes encodings that produce small difference for same faces and large difference for different faces
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_siamese_learning_goal.png)

## Triplet Loss

- look at one Anchor (A) image per person
- anchor compared to Positive (P)/matching image should have small difference
- anchor compared to Negative (N)/non-matching image should have big difference
- to make sure NN doesn't just learn all zeros
  - make sure difference is <= alpha (not zero) (alpha also called margin)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_FaceNet.png)
- triplet loss function:
  - NN doesn't care if loss is <= 0... just marks it as 0
  - Loss(Anchor, Positive example, Negative example)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_triplet_loss.png)
- if A and N are chosen randomly, it's easy to train
  --> choose A, P, and N that are hard to discern between
  - this increases computational efficiency of training
  - too many triplets would be really easy to train if A, P, and N are randomly chosen
- details in FaceNet
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_triplet_selection.png)
- a lot of companies release pretrained model weights trained on 10M-100M images,f or example

## Face Verification and Binary Classification

- as an alternative to triplet Loss function, you can treat similiarity function as binary classification
- loss could be: sum of logistic regression weights * (elementwise subtract image 1 FC layer from image 2 FC layer and take absolute vale) + beta constants
- or could use Chi Squared Similarity for loss function
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_similarity_function.png)

## What is neural style transfer?

- Generate image in same style as Van Gogh, say
- Content (C) + Style (S) --> Generated Image (G)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer.png)

## What are deep ConvNets learning?

- 9 representative neurons and 9 image patches that maximize their activations from each
- Layer 1 images:
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_1_images.png)
- deeper layers see more of the image
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_1_images_big.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_2_images_big.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_3_images_big.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_4_images_big.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_layer_5_images_big.png)

## Cost Function

- cost function for neural style transfer
  - how good of a generated image is it?
  - how similar is content of G to content of C?
  - how similar is style of G to style of S?
  - with hyperparameters alpha and beta
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_cost.png)
- G starts as random, then you minimize cost/J to iteratively get better image
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_gradient.png)

## Content Cost Function

- hidden layer l is usually somewhere in the middle for content cost
  - is l was too early on in NN, content would have to match original too closely
  - in contrast, i l was at the end, content would have to match too loosely
- use a pre-trained ConvNet (e.g. VGG)
- take: elementwise squared difference bw C and G in layer l / 2 (L2 norm)... elementwise Sum of Squared differences
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_content_cost.png)

## Style Cost Function

- in a given layer, how "correlated" are G's activations across channels 
  - really unnormalized cross covariance
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_style_cost_intro.png)
- but why does this capture style?
- which high level features (vertical texture, orange tint, etc) occur together
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_style_cost_intuition.png)
- Style Matrix
  - multiply elementwise across channels in G
  - if elements are both high, result is high; and vv if elements are both low
  - outputs matrix of size n_channels x n_channels
  - "gram matrix" in a math textbook
- compute Style Matrix for S and G both
- take the sum of squares of elementwise differences between the style matrices of S and G
- constant multiplier isn't that important bc you're using a constant as a hyperparameter anyway
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_style_cost_style_matrix.png)
- can do this across different layers to incorporate both low and high level features in style cost function
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_neural_style_transfer_style_cost_function.png)

## 1D and 3D Generalizations

- apply Convolution learnings to 1D or 3D data (non-images)
- 1D (e.g. time-series)
  - convolution applies to varies parts of time-series data
  - similar shape rules of convolution, channel, number of filters
  - ppl often use recurrent NN's in this cas
  - with 16 filters...
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_1d_convnet.png)
- 3D (Ct scan, movie data across time)
  - has HxWxDepth
  - H, W and D can be different (doesn't have to be a square/cube)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/4_ConvolutionalNeuralNetworks/week4/4wk4_3d_convnet.png)








