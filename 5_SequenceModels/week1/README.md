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

- this video inspired by the unreasonable effectiveness of RNNs
- many-to-many architecture where len(X) matches len(y)
- sentiment classification: many-to-one architecture
- one-to-one architecture: standard NN
- one-to-many architecture: music generation
- many-to-many: input and output lengths can be different
  - machine translation
  - for a sentence, output a y array output of a translation
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_RNN_architectures.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_RNN_architectures2.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_RNN_architectures_summary.png)
- there are some subtleties to one-to-many setup

## Language model and sequence generation

- RNNs are really good at this
- what is the probability of the sentence? chooses most likely sentence in speech recognition
- basic job is to input a sentence and estimate the probability of that sequence of words
- steps
  - tokenize words
  - map to one-hot vectors
  - add EOS token (end-of-sentence token)
  - might ignore punctuation, but could include it too
- what to do if word is not in dictionary
  - use <UNK> token
- at each step, RNN predicts probability of that word
  - at first step, RNN is predicting probability the sentence starts with a given word
  - at second step, give RNN the correct first word
  - at step n, give RNN the correct preceeding words 
- cost function
  - take softmax loss at each step, and sum them up to get the Loss 
- chance of entire sentence is multiplying out all probabilities output across all timesteps
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_language_model.png)

## Sampling novel sequences

- can be fun 
- informally get a sense of what your model has learned
- randomly sample from softmax distribution at activation layer 1
- input chosen word from previous layer into next layer
- stop when EOS token is reached, or if not included, just cap it at a number of words
- sometimes, it can generate <UNK> token
    - can just remove <UNK> if you want and choose another word instead
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_language_model_sampling_randomly.png)
- character-level RNN
  - sequence would be characters instead of words
  - can include caps and lowercase
  - pros: never have to worry about unknown word tokens
  - cons: main disadvantage is much longer sequences, so they're not as good at capturing long-range dependencies
- mostly word-level representations are used
- as computational power increases, more specialized applications are using character level representations more and more
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_language_model_char_level.png)
- way to see kinda how well it's working
  
## Vanishing gradients with RNNs

- basic RNN runs into vanishing gradient problems
- language can have very long-term dependencies
  - basic RNN not good at handling these, bc of vanishing/exploding gradient problem
- exploding gradients might be easier to spot
  - implement gradient clipping (max allowable value)
  - relatively robust solution for exploding grads
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_vanishing_grad.png)

## Gated Recurrent Unit (GRU)

- variant of basic RNN 
- helps with longer-term dependencies
- helps with vanishing gradient problem
- basic RNN
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_basic_RNN.png)
- GRU
  - c a new variable - memory cell
  - at time t, memory cell will have c<sup>t</sup> value
  - c<sup>t</sup> = a<sup>t</sup> (same in GRU, different in LStM)
- consider overwriting c<sup>t</sup> with c_tilde<sup>t</sup>
  - c_tilde<sup>t</sup> is a candidate for replacing c_t
- update gate is a value bw 0 and 1
  - key idea of GRU
  - gamma_u
  - for intuition, think of gamma_u as always 0 or 1 (although, in practice, it's the result of a sigmoid function)
  - in most ranges, sigmoid is either very close to 0 or very close to 1
  - say, gamma_u is 1 if pluraal or 0 if singular
  - gate's job is to know when to remember or forget it's memory
    - if it's remembering subject of a sentence, when to let go of that memory
    - the CAt .... was full. <-- forget after was for singular/pluraral (knowing whether or not to use were or was)
  - memorizing if "cat" was singular or not
  - Ng thinks equations are easier to understand than infographics
- remarkably good at updating memory cell at the appropriate point
- bc gamma can be so close to zero, it doesn't suffer from vanishing gradient
- long-range dependencies possible
- c_t can be a vector, gamma, and c_tilde_t
  - can choose to keep some bits constant and update other bits in a matrix of c
  - so is updating and not updating some units at every step
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_basic_RNN.png)
- Full GRU
  - just one new variable
  - gamma_r - this gate gamma r tells you how relevant is c<t> minus one to computing the next candidate for c<t>
  - gamma_r is best setup after many many years of research of the design of these units
  - GRU is one of most commonly used units
    - researchers have converged to this
    - standard, common used in practice
    - pretty robust
    - other common unit is LStM
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_full_GRU.png)
