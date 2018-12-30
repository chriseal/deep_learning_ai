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
  
## how to set up Train/dev/test distributions

- make sure dev and test sets come from the same distribution
  - e.g. dev set is US, test set is GB
  --> randomly shuffle data

## Size of the dev and test sets

- traditional:
  - 70% train, 30% dev
  - 60% train, 20% dev, 20% test
  - not unreasonable with datasets <= 10K in size
- big data, with m = 1M
  - 98% train, 1% dev, 1% test
- size of test set should be big enough to give high confidence in the overall performance of your system
- not having a test set would be possible (though Ng finds it reassuring to have test set), but not totally unreasonable if dev set is big enough

## When to change dev/test sets and metrics

- changing evaluation metric midway through a project
- add weight term to change importance of weight of certain examples (e.g. high weight for pornographic images)
- don't keep coasting with an incorrect evaluation metric
1 - define the target / evaluation metric
2 - how to aim at the target, cost or loss
- a model could do better on dev/test set, but worse in production if the two aren't from same distribution 
  - example: high quality images in training data, production has lower quality web uploads
- Ng: even if you can't define perfect evaluation metric, still come up with one and you can change it later  , because it helps so much to improve iteration speed


