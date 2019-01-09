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
