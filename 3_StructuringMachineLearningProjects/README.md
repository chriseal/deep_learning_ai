# Structuring Machine Learning Projects 
# Week 3

## Orthogonalization

- best ML engineers do one thing at a time
- tv's designed so each knob only does one thing at a time, cars too (not changing direction and speed of car at same time)
- orthogonal controls make machines easier to operate
- Chain of assumptions in ML
  - fit training set well -> fit dev set well -> fit test set well -> perform well in real world

## Single number evaluation metric

- quickly tell if the thing you tried works better or not
- at the beginning of a project, decide on single, real-valued metric to assess model 
- for example, use F1-Score = 2 / (1/Precision + 1/Recall) 
  - harmonic mean of precision and recall
  
## Satisficing and Optimizing metric

- what if you care about accuracy and running time?
  - hard to really combine these two in one metric
  --> maximize accuracy subject to running time <= 100ms, a Satisficing metric
- optimizing metric: really care about
- satisficing metric: can define as a binary that you care about, i.e. a threshold
- trigger words in Alexa
  - care about accuracy and false positives
  --> maximize accuracy, at most 1 FP every 24H
  
  
