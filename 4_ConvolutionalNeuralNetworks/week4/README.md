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







