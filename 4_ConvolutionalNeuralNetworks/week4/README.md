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






