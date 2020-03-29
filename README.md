# Interference

## [Author Verification Using Common N-Gram Profiles of Text Documents](https://www.aclweb.org/anthology/C14-1038.pdf)
The formulas form the [presentation](https://docs.google.com/presentation/d/1BZhBRqKzosFH2LZMjeQsJ-l_2NAoIszGsNeXn3zk0Z8/edit#slide=id.p) are duplicated in the Class implementation [notebook]((https://github.com/ovbystrova/Interference/blob/master/Class.ipynb)).


### Participants:
- Bystrova Olga [(ovbystrova)](https://github.com/ovbystrova) 
- Okhapkia Anna [(eischaire)](https://github.com/eischaire)
- Ryazanskaya Galina [(flying-bear)](https://github.com/flying-bear)


---
## The tasks
Links in the text lead to the notbooks where the mentioned task is done.
### Objective
In the original article the authors had interinsic authorship attribution task as a binary classification: the text was written either by the same author or by someone else. We could have simulated this structure using language background (*LB*) and first (native) language (*FL*) as "authors". However, it would not be ecologically valid, as the texts are, of course, written by different authors, and we did not have the data on authorship. Thus, we changed the task to be binary (for LB) and multiclass (for FL) classification.

### Pipeline
1. [preprocessing](https://github.com/ovbystrova/Interference/blob/master/JSON_Files.ipynb)
  1. tokenization for word n-grams (of length n)
  2. truncation so that all texts are of the same length (omitting the shorter texts)
  3. train/test split  (correcting for imbalanced classes!)
    1. on FL, native language
    2. on LB, speaker type
3. building classifiers [for each parameter combination](https://github.com/ovbystrova/Interference/blob/master/Class.ipynb)
  1. calculation of n-gram profiles (P)
  2. cutoff of the most frequent L
  3. distance calculation
4. multiclass classification with minimal distance for each ensemble, averaging the results
    1. on FL, [native language](https://github.com/ovbystrova/Interference/blob/master/Language_Testing.ipynb)
    2. on LB, [speaker type](https://github.com/ovbystrova/Interference/blob/master/LB_Testing.ipynb)
5. building [baselines](https://github.com/ovbystrova/Interference/blob/master/Baseline.ipynb)
  1. TF-IDF + logistic regression
  2. TF-IDF on word bigrams + logistic regression with parameter search
  3. word2vec + logistic regression with parameter search
  4. word2vec + perceptron 
6. [comparing results](https://github.com/ovbystrova/Interference/blob/master/Report.ipynb)

### Architectural choices
- We decided to onbly use ensemble classifiers as they performed the best in the article.
- We decided to cut all the texts to the length of mode length and omit all texts shorter than that.
- We decided that we need to balance classes and select the same number of texts from each class, landing on two options - 90 and 400 from each class. All the classes with less datapoints were omitted.
- Character ensembles were slow and thus were only calculated for LB.
- We decided to only use the number of n-grams (L) to determine the length of a profile and to use multiclass classification with minimal distance, that does not need a threshold (θ). The parameteres from the original article (the ones we included in bold):

#### **Parameter space**
- size of N-grams (n)
    - **from 3 to 10 for characters**
    - **from 1 to 3 for words**
- size of a profile 
    - **Number of n-grams (L) 200, 500, 1000, 1500, 2000, 2500, 3000**
    - Fraction of n-grams from the shortest text (f) from 0.2 to 1 (increments of 0.1)
- Threshold (θ)
  - if more than 1 known-author document available (θ2+)
  - if only 1 known-author document available (θ1)
- **Ensemble size and parameters**

## Results
### On test
![test](https://github.com/ovbystrova/Interference/raw/master/data/on_test.png)
### On train (among radius distance models)
![train](https://github.com/ovbystrova/Interference/raw/master/data/on_train.png)
#### Only FL
![fl](https://github.com/ovbystrova/Interference/raw/master/data/fl_only.png)
#### Only LB
![lb](https://github.com/ovbystrova/Interference/raw/master/data/lb_only.png)

## Discussion
One can see that in ALL cases the simplest baseline model (TF-IDF + logistic regression) outperforms all others.  It is interesting that on of the radius distance models outperforms NN on language background, as NN shows bad results on LB. Another thing to notice is that charachter models are outperformed by word models on train, but not on test. Generally, longer n-grams yeild better results, but the rule also holds more true on train than on test.

The question is why does the radius distance is outperformed by the baseline, the simplest of the models? One could argue it is due to the method being unaplicable for multiclass classification, and being specifically created for intrinsic authorship attribution. 

There is another issue connected to this unapplicability. Training each radius distance model took A LOT of time (up to 5 hours) while training logisitic regression and even simple NN took almost no time (under 5 finutes). This is one of the limitations of the radius distance algorithm, as it's complexity and thus time is proportional to the number of distance calculations. This, in turn, is proportional to (1) the number of classes, (2) the number of texts in each class, (3) the profile length. In the article the number of classes was 2 and the number of texts was below 50, which made the time aspect unimportant.

The concusion is that the method might be well-suited for intrinsic authorship attribution, but not for extrinsic authorship attribution, which is essentially multiclass classification that we had.
