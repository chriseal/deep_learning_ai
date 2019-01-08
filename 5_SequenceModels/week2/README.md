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
