# Structuring Machine Learning Projects
# Week 2

## Carrying out error analysis

- whether or not to improve model that misclassifies cats as dogs; model has 10% error
- steps:
  - take 100 misclassified files
  - count number of dog pics
  - if only 5 pics of dogs that are misclassified, then only 5% of misclassified are dogs
    - this is the ceiling for how much better your model could be with making it perfect on dog pics
    - ceiling goes from 10% to 9.5%
  - but if 50 pics are dogs, then your ceiling goes from 10% error to 5% error and it's probably a worthwhile effort
- evaluate multiple ideas in parallel
  - create a table/spreadsheet that counts up fraction of mislabeled examples of each case and make notes 
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/3_StructuringMachineLearningProjects/week2/3wk2_error_analysis_spreadsheet.png)

## Cleaning up incorrectly labeled data

- if Y is incorrect for some examples (label is a dog and it should be a cat)
- DL algorithms are quite robust to **random** errors in the training set
  - if mislabeled examples are random, might be okay to leave it alone
- DL algorithms are affected by **systematic** errors in training set
- mislabeled examples in dev or train set
  - add "incorrectly labeled" column to error analysis sheet
  - how to decide if it's worth your time to correct
    - 10% error on dev set, 6% mislabeled -> 0.6% error due to incorrect labels (not worth your time)
    - 2% error on dev set, 6% mislabeled -> 1.4% error due to incorrect labels (worth your time)
    - or one model has 2.1% error and another model has 1.9% error and 0.6% error due to mislabels so you can't tell which model is better
- how to fix incorrectly labeled dev/test set examples
  - do same process for dev and test sets (make sure they're from same distribution)
  - consider examples model got **right** and ones it got wrong (to be fair)
  - possibly correct labels in training set, but might be okay to skip re-labeling this
    - if training set comes from slightly different distribution than 
- very important to manually review examples, in order to determine priority for improvementa

## Build your first system quickly, then iterate

- lots of directions you could go
- build initial system quickly 
  - a lot of the value of the initial system is that it gives you a way to conduct error analysis
- use Bias/Variance analysis and Error analysis to prioritize next steps
- most teams overthink initial solution
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/3_StructuringMachineLearningProjects/week2/3wk2_speech_detection_error_analysis_example.png)

## Training and testing on different distributions

- because models are hungry for tons of data, more often, people are training on a different distribution than testing on
- if you can get 200K examples from webcrawling and 10K from mobile apps, but your model is for predicting images in mobile apps
  - Option 1: combine both datasets, shuffle them, and distribute into train, dev, test
    - major drawback: aiming for mostly webcrawling distribution, which isn't what you want
  - Option 2 (recommended):
    - train - 200K from webcrawling, 5K from mobile app
    - dev, test - 2.5K from mobile app
    - aiming in the right place is what you want
- Example: speech recognition for cars
  - training: purchased data, smart speaker control, voice keyboard -> 500K utterances
  - dev/test: 20K speech activated rear view mirror (use case)
  --> training - 500K+10K
  --> dev/test - each 5K of use case data
  
## Bias and Variance with mismatched data distributions

- estimating Bias and Variance changes when train and dev/test are from different distributions
- maybe dev set is just much harder to predict if train is cleaner data
- could higher error in dev set 
- when going from training error to dev error, two things change at a time
  - 1) algorithm saw data in training data but not in dev set (variance)
  - 2) distribution of train and dev data is different
- create training_dev set: same distribution as training set, but not used for training
  - so train, training_dev (same distribution) and dev, test (different distribution)
  - this helps determine if you have a bias or variance problem
- data mismatch problem:
  - model does well on train, training_dev but poorly on dev <-- data mismatch error
- Avoidable Bias, Variance and Data Mismatch to take into account now
![img](https://github.com/chriseal/deep_learning_ai/blob/master/3_StructuringMachineLearningProjects/week2/3wk2_bias_variance_mismatch_sheet.png)
- filling out 2 top right cells can give you additional insight (image below)

## Addressing data mismatch

- what if you have a data mismatch problem? no simple way to know what to do next
- carry out manual error analysis to try to understand how train and dev are different
  - e.g. train is conversational speech, dev is addresses and directions (more often)
- make training data more similar to dev set
  - data synthesis
    - e.g. add clean speech to car noise and add reverb to synthesize data inside the car
    - what if you have 1H of car noise and 10000H of speech?
      - possible to overfit to 1H of car noise if you repeat it 10,000 times
    - e.g. could synthesize pictures of cars using visual rendering systems
      - possible to overfit to the way the cars are rendered
      - same issue if you got car images from video games
 - speech recognition, Ng: "I've seen artificial data synthesis significantly boost the performance of what were already very good speech recognition system"
