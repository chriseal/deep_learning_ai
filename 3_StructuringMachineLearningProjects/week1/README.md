# Structuring Machine Learning Projects
# Week 1

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

## Why compare to human-level performance?

- Bayes Optimal Error: best possible error, best possible function to map from X to Y
- performance improvements slow down after surpassing human level performance
  1 - human level performance is not that far from Bayes Optimal Error
  2 - tools to use to improve performance are less after performance surpasses human level performance
    - get labeled data from humans
    - gain insight from manual error analysis: why did this person get it right?
    - better bias/variance analysis

## Avoidable bias

- say, human error is 1%, dev is 10%, training error is 8% (avoidable bias is 7%) --> try to reduce bias
- say, human error is 7.5%, dev is 10%, training error is 8% (avoidable bias is 0.5%) --> try to reduce variance to bring dev error closer to train error
- human level error used as a proxy for Bayes error
  - works for image classification
- *Avoidable Bias*: difference between training error and Bayes error
- Variance: difference between training error and dev error

## Understanding human-level performance

- human level error can be a proxy for Bayes error
- what is human level error ? laymen, experts, team of experts?
  - theoretical optimum is team of experts, so that's human level performance

## Surpassing human-level performance

- problems where ML surpasses humans
  - online advertising placements
  - product recommendations
  - logistics transit time
  - loan approval
- not natural perception tasks (NLP, image recognition),
- structured data
- lots of data (more than human can examine)
- some natural perception tasks that have surpassed humans
  - some medical radiology
  - some speech recognition
  - some image recognition

## Improving your model performance

1 - fit training set pretty well
  - Avoidable bias - bigger network, longer/better optimization algorithms (RMS Prop, Adam), better architecture/hyperparameters
2 - training set performances generalizes well to dev/test set
  - variance - regularization (L2, dropout, data augmentation), more data, NN architecture
- systematically improve performance of model (rather than shotgun approach)
![img](https://github.com/chriseal/deep_learning_ai/blob/master/3_StructuringMachineLearningProjects/week1/3wk1_toolbox_to_improve.png)
