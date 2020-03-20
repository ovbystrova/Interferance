import numpy as np

def softmax(scores, inverse=True):
  """
  normilizes class scores, making them correspond to probability, adding up to 1

  :param scores: list of int or float, scores
  :inverse: bool, if lower scores indicate higher probability,
            optional, default True
  :return: list of int or float, normilized probability-like scores
  """
  if inverse:
    scores = [-1*score for score in scores]
  normalizing_constant = np.sum(scores)
  normalized_scores = [score/normalizing_constant for score in scores]
  return normalized_scores