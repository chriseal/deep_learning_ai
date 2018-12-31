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

## 
