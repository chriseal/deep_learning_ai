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
  - create a dictionary (30-50K common, 100K not uncommon)
  
