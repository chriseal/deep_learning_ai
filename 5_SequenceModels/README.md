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

## Long Short Term Memory (LSTM)

- even more powerful than GRU, slightly so
- more general version than GRU
- LStM vs GRU
  - a_t != c_t
  - gamma_u gets replaced by gamma_f (forget gate) and gamma_u
    - can keep the old value and just add to it if it wants to
  - new output gate: gamma_o
  - three gates instead of two
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_gru_vs_lstm.png)
- LStM in pictures
  - Ng thinks equations easier to understand than pics
  - relatively easy for c_0 to be passed further down to the right
  - can add "peephole connection" to gate calculations to effect gate value indepently
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_lstm_pic.png)
- no one model is best yet
  - GRU is simpler model - computationally faster, scales to bigger networks slightly more easily
  - LStM model: more powerful, more flexible, slightly slower
    - historically more proven choice, default first thing to try 
    - in recent years, GRUs are growing and more teams are trying them, bc simpler with just as good of results

## Bidirectional RNN

- gets information from past and future
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_teddy.png)
- Bi-RNN with 4 sentence input, example
  - defines an acyclic graph
  - forward propogation that goes backwards (right to left) too
  - can be GRU or LStM can be used
  - LStM is most common with NLP BRNN tasks
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_brnn.png)
- basic change to GRU or LStM model where you include all events in sequence
- pretty effective if you have full sequence
- disadvantage: need full sequence bf making predictions
  - need entire utterance bf making speech recognition model (for standard BRNN, more complex models are used that incorporate BRNN's somehow)

## Deep RNNs

- Deep RNN with 3 hidden layers, example
  - each box in this chart has two inputs
  - unique set of parameters shared across each layer, but unique across layers
  - deep RNN's having 3 layers is already quite a lot
    - RNN's not usually 100 layers etc
    - but could have deep network that isn't RNN tagged onto each y output
- each block can be GRU or LStM
- can have BRNN's in there
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week1/5wk1_deep_rnn.png)


# Sequence Models - Week 2 - NLP

## Word Representation

- can learn synomyms e.g.
- one hot representation treats each word independently
  - all relationships are independent (doesn't know juice often occurs after both apple and orange often)
- featurized representation
  - apple and orange have more similar vectors when featurized
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_featurized_word_embeddings.png)
- **t-SNE**
  - common way to visualize 300Dims into 2: 
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_tsne.png)
- word gets "embedded" to a unique point in 300 Dimensional space, hence the word "embedding"   

## Using word embeddings

- how to use word embeddings in an application
- embeddings makes it easier to train
  - durian is rare type of fruit
  - cultivator is similar to farmer
  - how to discover orange and durian are both fruits
  - embeddings learned on 1B to 100B words
  - training set for your task may have much smaller number of words to train on
    --> transfer learning of word embeddings
  - bi-directional RNN better for NE recognition
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_NE_rec_ex.png)
- can now use 300dimensional vector (dense) rather than 10K-100K+ vector (sparse) to represent a word
- can optionally fine-tune word embeddings
  - only if you have a pretty large dataset on hand
- less useful to machine translation, language modeling but very useful for everything else\
- similar-ish to face recognition
  - siamese network --> 128D encoding/embedding of faces
  - one difference: take as input any image, in word embeddings there is a fixed number of words defined by vocabulary
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_face_rec.png)

## Properties of word embeddings

- analogies are important aspect of word embeddings
- take elementwise difference bw wdEmb's
- when queen is input into equation, lHand side more closely matches RHand side
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_analogies.png)
- formalizing the implementation...
- find word that maximizes similarity bw emb_x and e_king-e_man+e_woman
- remarkable thing is this actually works
- 30-75% accuracy in analogies in research paper, counting only if you get the exact word right
- parallelogram relationships in analogies only work in 300D space..t-sne's non-linear transformation doesn't make it a reliable way to view analogies
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_analogies_explained.png)
- Cosine Similarity:
  - numerator: inner product bw two vectors
    - if they're similar, the numerator will be large
  - angle of 0: cosine = 1
  - angle of 90: cosine = 0.5
  - angle of 180 (opposite): cosine = 0
- can also use Euclidean distance, but it's not used as much as cosine similarity
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_cosine_similarity.png)

## Embedding matrix

- 300 x 10K (or vocab size)
- in practice, you'd use a function to pull the appropriate column, since matmul is less efficient
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_embedding_matrix.png)

## Learning word embeddings

- used to have complicated models, but now, simpler models work if you have a lot of data
- neural language model
  - earlier model and pretty effective for word embeddings
  - reasonable way to learn word embeddings
  - moving window as a parameter --> only look at last 4 words
    - 300*4 = 1200 length feature vector input into model
  - need similar features for similar words to predict what the next word is - e.g. "apple *juice*", "orange *juice*"
  - context: last 4 words
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_neural_language_model.png)
- if goal isn't a language model, you can change your context
  - context: 4 words on L and 4 words on R, input concatenated embeddings into NN 
  - context: 1 word on L
  - context: nearby 1 word, works surprisingly well
    - skip-gram model
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_other_contexts.png)

## Word2Vec

- thomas Mikolov
- **SkipGram model**
- randomly pick a word that's within +/- n (5,10,etc) words before or after
- goal isn't to do well on the supervised learning 
- theta_t is probability of word t
- matrix E has a lot of parameters
- problem:
  - computational speed, if vocab is 10K, summing over denominator is very slow 
  - even harder to increase vocab size
- hierarchical softmax
  - fixes computational cost 
  - split vocab into halves in each node in a tree
  - computational cost scales at log(vocab_size) rather than linearly with vocab size
  - doesn't use a symmetric split at every node
  - bury rare words deeper in the tree
  - most common words higher up in the tree
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_hierarchical_softmax.png)
- negative sampling 
  - simpler than hierarchical softmax
- how to sample context c
  - reduce sampling of frequently occuring words
  - increase sampling of rarer words
  - various hieristics to improve sampling
- Skip-Gram and CBOW (continuous bag of words, takes surrounding words)
        
## Negative Sampling

- defining a new learning problem
  - sample a context of words randomly from a dictionary/vocab as incorrect words
    - okay if randomly selected words come from context area around original word
  - correct word is target word
  - define correct as 1's and 0's for incorrect target words
  - turn into a supervised problem
  - k is 5-20 for smaller datasets
  - k is 2-5 for larger datasets
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_generating_supervised.png)
- K-to-1 ratio of negative to positive examples
- instead of calculating binary cost at each word in vocab, for each prediction
  - you are instead calculating loss for 1+k binary cost calculations
- this technique is called negative sampling
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_negative_sampling.png)
- but how to sample negative words?
  - uniformly random, sampling in ratio of usage <-- doesn't work
  - take frequency of occurence of word_i to the 3/4 is what worked best empirically in the study
  - this is somewhere between uniformly random sampling and sampling in ratio of usage
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_how_to_sample.png)

## GloVe word vectors (global vectors for word representation)

- GloVe algorithm has some enthusiasm
- not used quite as much as skip-gram or word2vec algorithms, but still common
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_GloVe.png)
- x_ij counts co-occurences of i and j in context of each other (however you define context)
- minimize difference bw...
  - theta and e_j terms measure how related two words are given the co-occurence in context
  - f(x_ij) is weighting term = 0 when no co-occurences
  - "0 * log 0" = 0
  - weighting factor gives reasonable weight to rare words
  - gives frequent words not too much weight
  - e and theta are symmetric
  - final e is mean of e and theta for a given word
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_GloVe_cost.png)
- individual dimensions may not be interpretable (cannot guarantee interpretability)
- can't guarantee that axis used to represent the features will be well-aligned with what would be easily interpretable
  - not necessarily orthogonal axes
  - but parallelogram map for figuring out analogies still works

## Sentiment Classification

- with word embeddings, you're able to build good sentiment classifiers even with only modest-size label training sets
- may not have a large training set
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_sentiment_training_set.png)
- can apply embeddings to words that weren't in your training / labeled dataset
- you can take the average or some of the word embeddings of the text
- works for reviews that are short or large
- con: ignores word order "completely lacking in *good* service"
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_sentiment_simple_mod.png)
- instead of averaging, you can use a many-to-one RNN
  - input word embeddings as x's
  - since word embeddings use a large dataset, this algorithm will generalize better bc of transferred knowledge
  - can still deal with words that weren't in your traning set, but were in embeddings (that you stole)
  
## Debiasing word embeddings

- embeddings pick up the biases in text used to train the model
- more easy to reduce bias in AI than reduce bias in human race
1. identify bias direction
  - take a few differences and average them
  - this tells you the bias direction/axis
2. for every word that is not definitional (e.g. unrelated to gender), project along axis to get rid of bias
  - how do you decide which words to use?
  - train a classifier to determine which words should be gender specific (most words are not definitional in english)
3. equalize pairs
  - e.g. make grandmother and grandfather equidistant from babysitter
  - relatively small number of pairs, possible to handpick pairs you want to neutralize
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week2/5wk2_addressing_bias.png)


# Sequence Models - Week 3 - Sequence-to-Sequence Models

## Basic Models

- e.g. Machine translation
  - encoder network for original language x 
    - produces encoding / embedding of original sentence
  - feed this encoded sentence into a decoder network to then make the translation
  - this actually works decently well
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_encoder_decoder.png)
- Image captioning
  - can use CNN for decoder
  - use RNN for encoder
  - works pretty well, esp. if captioning is not too long
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_image_captioning.png)
- but you don't want a randomly chosen caption or translation, you want the most likely one...

## Picking the most likely sentence

- decoder model is very similar to language model, except that instead of starting with an array of all zeros for x, it starts with a decoded array (green in img below)
- Probability of English sentence conditioned on an input French sentence
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_conditional_machine_translation.png)
- need to maximize likelihood of probabilities to get the most probable sentence
  - i.e. don't randomly sample from the probability distribution; otherwise, you'd end up with some okay but imperfect translations and possibly some bad translations
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_conditional_machine_translation2.png)
- Why not just use a greedy search?
  - i.e. choose most likely first word, then most likely second word, etc
  - i.e. maximizing joint probability
  - chance of "going" higher than chance of "visiting" (bc going is more common word)
  - but this ends up in less optimal translation than the first example in the img below
  - (argmax portion of this image added to describe what approximate searches try to do)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_greedy_search.png)
- most common search method is an approximate search that finds argmax-ish of total sentence

## Beam Search

- say, you have 10K vocab
- beam width parameter
  - set it to, say, 3
  - adds 3 most likely words for each token
  - keep these 3 words in memory
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_width.png)
- then, for translated_word_2, it will consider all 3/beam_width previous words from translated_word_1
  - find pair of tw1 and tw2 that is most likely given word 1 and french sentence
  - do this for all 3/beam_width previous words
  - 3 * 10k (i.e. vocab_size) comparisons made, and you choose the 3/beam_width best
  - this can cancel out possible choices for first word (e.g. image doesn't choose September as first word since it's not included in 3 best possibilities for tw2)
  - instantiate 3 copies of the network to search through 10K/vocab_size words at each copy (no need to make 30K copies of the network!)
  - x in the image is the entire French sentence (not just 1 word)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_search.png)
- terminates with EOS tag
- if Beam_width = 1, it will become greedy search discussed above
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_search2.png)

## Refinements to Beam Search

- Length normalization
  - small change that can get much better results
  - numerical underflow an issue by raw multiplication of a lot of small numbers
  - in practice, we take logs
  - the log of a product becomes the sum of the logs
  --> numerically stable algorithm less prone to rounding errors
  - maximizing log should be the same as maximizing non-log, since log is monotonically increasing
  - without length normalization, probabilities of a short sentence is higher than longer sentence (log or no log) so w-out normalization the algorithm unnaturally prefers shorter sentences
  --> normalize probability by length of sentence
  - divide by t_y<sup>alpha</sup> where alpha is a parameter
  - alpha is a hack/hieuristic that works well in practice... try a few in practice and see what works best
  - you see a lot of sentences with length 1,2,3 and run it all the way up to 30 length
  - look at all output sentences and score them with cost function on the bottom of the img
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_length_normalization.png)
- choosing beam width length
  - performance vs computational cost trade off
  - common to see beam width of 10 in production
  - 100 high, but seen in some production systems
  - 1000 for some some research applications
  - diminishing returns: huge gains going 1-10, less so higher up
  - beam search isn't guaranteed to find exact maximum, but runs much faster than bread first search or depth first search
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_width_param_choosing.png)
- error analysis is one of most important ways of improving algorithm (some simple things can help)

## Error analysis in beam search

- how do determine if beam search or RNN is causing errors?
- always tempting to increase beam width
- compare probabilities computed by RNN to probabilities computed by beam search to determine which was bigger in order to determine next steps / ascribe particular error to RNN or beam search
- y<sup>*</sup> is human translation, y<sup>^</sup> is beam search y
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis.png)
- case 1: if prob human translation is higher than beam search prob, beam search is at fault
- case 2: if beam search prob is gte human translation prob, RNN is at fault
- need to take into account length normalization to do this properly
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis2.png)
- go through dev set and find the mistakes, determining which RNN or BS is at fault
- carry out error analysis to figure out fraction of errors due to RNN (or BS)
- only if you find BS is responsible for most of the errors do you increase beam width
- in contrast, if RNN is at fault, do a deeper error analysis related to RNN (see Week 3)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis3.png)

## Bleu Score (optional)

- multiple equally good answers in machine translation
- how do you evaluate machine translation system if there are multiple acceptable/equally good answers
- Bleu score (Bilingual evaluation understudy)
  - automatically computes a score of a translation
  - as long as translation is close to a human generated reference (as part of dev/test sets)
  - "understudy" for having humans review every translation
  - precision: if machine translation in human refs, what fraction of words that you predicted are in refs
  - modified precision: "the" gets credit up to the max number of times "the" is in one of the human generated refs
- bigrams portion of blue score:
  - compute modified precision for bigrams
  - sum of matching bigrams / sum of bigrams in machine translation
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score.png)
- perfect score is exactly matching translation
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score2.png)
- BP = brevity penalty
  - penalizes if machine translation length is less than human translation length
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score3.png)
- useful single evaluation metric --> moved progress forward
- not used in speech recognition (bc only 1 right answer)
- used in machine translation and image captioning

## Attention Model Intuition

- works better than encoder / decoder
- very important idea in ML
- encoder / decoder "memorizes" entire sentence and then translates/decodes it from scratch
- encoder / decoder works well for short sentences but is bad for longer sentences (hard for it to "memorize" it all)
- attention model is green line of performance below
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_issue.png)
- attention model is very influential paper - Bahdanau et al 2014 "neural machine translation by jointly learning ..."
- BRNN
- attention model computes attention weights
- in first word, you don't need to be looking way at the end of the sentence
- alpha<sup>(1,1)</sup>, alpha<sup>(1,2)</sup>....alpha<sup>(1,len_text)</sup>
- for first word, this combination of weights determines the context of the first word
- new alpha array / context for second word and so on
- activation depends on context of current row and state (s) of previous step
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_intuition.png)

## Attention Model

- attention model allows NN to pay attention to only part of the sentence much like how a human translator would
- a<sup\<6\></sup> for backprop is initialize as a vector of all zeros
- a<sup\<t'\></sup> is the concatenated of forward occurence activations and backward occurence activations for BRNN of French sentence (original sentence)
- alpha is the attention parameter, tells us how much the context will depend on the attention weights
- C is weighted sum of attention and activations
- alpha<sup>\<t, t'\></sup> is the amount of attention y<sup>\<t\></sup> should pay to a<sup>\<t'\></sup>
  - when generating output y_t, how much attention should you be paying to the t' input word? (alpha tells you this)
- alphas change for every input word
- state and context (top left) look like a pretty 'normal' RNN predicting one word at a time
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_model.png)
- for every fixed value of t, alphas sum to 1 (see denominator in a_tt' equation
- how to compute exponential values
  - one way is to use a small NN
  - only 1 hidden layer bc you calculate these a lot
  - leave details to backprop
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_computing_alphas.png)
- this actually works, learns to pay attention to the right things
- takes quadratic cost to run this algorithm
  - but in machine translation, most texts are pretty small
- also applied to image captioning
  - pays attention to parts of the picture at a time
- date normalization
- visualization of attention weights/alphas
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_examples.png)

## Speech recognition

- audio clip to transcript
- preprocessing - create a spectrogram from audio clip
  - how loud is this sound at different frequencies at different times
- no need for phonemes
  - in academia, 3000H of audio would be reasonable
  - in production systems, 100,000H of audio makes deep learning work better
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_speech_recognition_problem.png)
- CtC cost for speech recognition works well
  - Connectionist temporal classification
  - in practice, will be Bi-GRU/LStM
  - # input time steps is much larger than output
  - collapse repeated characters not separated by blank
  - skip blanks/underscores
  - NN has 1000 outputs in this example, but CtC reduces effective output to reasonable characters/words
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_ctc_cost.png)

## Trigger Word Detection

- put a few ones after when trigger word was said
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_trigger_word_detection.png)




