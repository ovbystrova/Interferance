import numpy as np

def distance(profile1, profile2):
  """
  calculates distance between 2 document profiles

  :param profile1: collections.Counter, token frequencies in profile 1
  :param profile2: collections.Counter, token frequencies in profile 2
  :return: float, distance
  """
  vocab = set(profile1.keys()).union(set(profile2.keys()))
  diffs = []
  for word in vocab:
    fp1 = profile1[word]
    fp2 = profile2[word]
    diff = ((fp1 - fp2)/((fp1 + fp2)/2)) ** 2
    diffs.append(diff)
  return np.sum(diffs)


def radius(di, u, A):
  """
  compares proximity between an unknown document and a known one
  :param di: collections.Counter, token frequencies in a known document i
  :param u: collections.Counter, token frequencies in an unknown document
  :param A: list of collections.Counter, token frequencies of all know documents
  :return: float, distance
  """
  return(distance(di, u)/np.max([distance(di, ai) for ai in A]))


def radius_distance(u, A):
  """
  compares proximity between an unknown document and all known ones
  :param u: collections.Counter, token frequencies in an unknown document
  :param A: list of collections.Counter, token frequencies of all know documents
  :return: float, distance
  """
  dists = []
  for di in A:
    dists.append(radius(di, u, A))
  return np.mean(dists)