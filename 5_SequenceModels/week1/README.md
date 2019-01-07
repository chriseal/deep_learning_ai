# Sequence Models - Week 1

## Why sequence models?

- from sequence to sequence
  - speech recognition - input: audio, output: text
  - music generation - input: null, output: music
  - DNA sequence, machine translation, named entity recognition, video 
  - x and y can have different lengths; only input or only output can be a sequence
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_examples_of_sequence_models.png)

## Notation

- Named entity recognition example
- x<sup>\<t\></sup> implies a temporal sequence (but we'll use it even if it isn't a temporal sequence)
- T_x : length of input sequence, (T_xi specific to ith training example)
- T_y : length of output sequence, (T_yi specific to ith training example)
- X<sup>(i)\<t\></sup> - t<sup>th</sup> element in i<sup>th</sup> training example
- y<sup>(i)\<t\></sup> - t<sup>th</sup> element in i<sup>th</sup> training label
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_notation.png)
- Representing words
  - create a dictionary (in production: 30-50K common, 100K not uncommon)
  - choose 10K most common words
  - use one-hot representation of each word (0s and 1s)
  - create token called unnamed word to represent words not in your vocabulary 
  
## Recurrent Neural Network Model

- why not a standard network?
  - doesn't work well
  - inputs and outputs can be different length
  - doesn't share features learned across different positions of text (harry in token1 versus token5898)
  - better representation can reduce number of parameters
- recurrent neural networks
  - reading sentence from left to right
  - take first word and try to predict output
  - when reading second word, instead of just predicting y_2, it gets y_1's activation value (or some information)
  - initialize a_0 randomly is most common choice for time zero activation
  - RNN - scans through data from L to R
  - parameters are shared for each timestamp 
  - image structure doesn't look forward and this is a problem (bi-directional RNN's fix this)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_structure.png)
- Forward Propogation
  - a_0 all zeros
  - W_ax <-- x means W is transforming some x-like quantity, a means W is being used to compute some a-like quantity
  - typically, a tanh is used as an activation, instead of ReLU (though ReLU is still used) as we have other ways of addressing vanishing gradient proglem
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_forward_propogation.png)
- simplifying the notation:
  - concatenate horizontally W_aa and W_ax
  - [...] is a vector of shape (10.1K, 1)
    - a_\<t-1\> is (100, 1) vector
    - x_\<t\> is (10K, 1) vector
  - W_a is a matrix of shape (100, 10.1K)
  - this compresses notation so we go from W_aa and W_ax / two parameter matrices, to W_a and a vector, so one parameter matrix
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_simplified_notation.png)

## Backpropagation through time

- this is often done automatically in programming frameworks
- W_a and b_a are same parameters used for every step
  - same for W_y and b_y
- can use any loss
  - categorical cross entropy
  - compute loss at every time step
  - total Loss is sum of loss at every timestep
  - most significant calculation for backprop is activation params
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_backprop_through_time.png)

## Different types of RNNs

- 
