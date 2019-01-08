# Sequence Models - Week 2

NLP

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


