# Interference

## [Author Verification Using Common N-Gram Profiles of Text Documents](https://www.aclweb.org/anthology/C14-1038.pdf)
[presentation](https://docs.google.com/presentation/d/1BZhBRqKzosFH2LZMjeQsJ-l_2NAoIszGsNeXn3zk0Z8/edit#slide=id.g7e294f0bb6_0_100)

### PIPELINE:
1. tokenization for word n-grams (of length n)
2. truncation so that all texts are of the same length (all but one model)
3. if only one known-author document given - split into halves
4. calculation of n-gram profiles (P)
5. cutoff of the most frequent L or fraction f from the profile
6. distance calculation - mean over r(di, u, A) for all di in A
7. threshold-based binary classification (θ)

### Options space:

- size of N-grams (n)
    - from 3 to 10 for characters
    - from 1 to 3 for words
- size of a profile 
    - Number of n-grams (L) 200, 500, 1000, 1500, 2000, 2500, 3000
    - Fraction of n-grams from the shortest text (f) from 0.2 to 1 (increments of 0.1)
- Threshold (θ)
  - if more than 1 known-author document available (θ2+)
  - if only 1 known-author document available (θ1)
- Ensemble size and parameters
