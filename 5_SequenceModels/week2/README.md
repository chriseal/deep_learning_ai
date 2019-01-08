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

