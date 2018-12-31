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
![img](https://github.com/chriseal/deep_learning_ai/blob/master/3_StructuringMachineLearningProjects/week1/3wk2_error_analysis_spreadsheet.png)

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


