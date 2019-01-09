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
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_greedy_search.png)
- most common search method is an approximate search that finds argmax-ish of total sentence
