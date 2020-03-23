import numpy as np
from tqdm import tqdm
from collections import Counter
from modules.single_classifier import SingleClassifier

def make_counter(commons):
    return Counter({x[0]: x[1] for x in commons})

class EnsambleClassifier():
    """
    classifies texts as belonging to several classes given true classes
    by combining several single classifiers

    :param profiles_multiple: list of list of collection.Counter,
                              n-gram profiles (of each n-gram type)
    :param y_true: list of str or int, true classes of each text from profiles
    :param p_length_options: itearble of int, options for length of
                             truncated profiles
    :param classes: list of int or str, possible classes

    Attributes:
      y_true, p_length_options, profiles_multiple, classes,
      classifiers: list of SingleClassifier, each parameter combination
    """


    def __init__(self, profiles_multiple, y_true, p_length_options, classes):
        self.y_true = y_true
        self.p_length_options = p_length_options
        self.profiles_multiple = profiles_multiple
        self.classes = classes
        self.classifiers = self.create_ensamble()

    def create_ensamble(self):
        """
        creates a SingleClassifier for each parameter combination

        :return: list of SingleClassifier, each parameter combination
        """
        classifiers = []
        for profiles in self.profiles_multiple:
            for p_length in self.p_length_options:
                classifiers.append(SingleClassifier(profiles, self.y_true, p_length, self.classes))
        return classifiers

    @staticmethod
    def majority_vote(votes):
        """
        majority voting,
          if several classes have equal number of votes
          returns the alphabetically first class

        :param votes: list of lsits str or int,
          predictions of each classifier on each object
        :return: list int or str, resulting prediction
        """
        votes = np.array(votes).T
        pred = [Counter(vote).most_common(1)[0][0] for vote in votes]
        return pred

    def confidence_summing(self, classifier_distances):
      """
      confidence voting, distances to classes are summed

      :param classifier_distances: list of list of dicts
                        {str or int: float or int},
                        confidences of each classifier on each object
      :return: list of dict {str or int: float or int},
              resulting confidences on each object
      """
      cummulative_distances = classifier_distances[0]
      if len(classifier_distances) == 1:
        return cummulative_distances
      else:
        for classifier in classifier_distances[1:]:
          for i, element in enumerate(classifier):
            for c in self.classes:
              cummulative_distances[i][c] += element[c]
        return cummulative_distances

    def weighted_vote(self, classifier_distances):
      """
      wheighted voting,
        distances to classes are summed for each object,
        minimal distance is the final predicted class

      :param classifier_distances: list of list of dict {str or int: float or int},
                        confidences of each classifier
      :return: list of int or str, resulting prediction on each object
      """
      sum_distances = self.confidence_summing(classifier_distances)
      return [min(class_dist, key=class_dist.get) for class_dist in sum_distances]


    def forward_ensamble(self, method='majority', confidence=False):
      """
      predicts classes using the ensemble

      :param method: str ('majority' or 'weight'), nethod of voting to use,
                     optional, default 'majority'
      :param confidence: bool, whether to return proximity score for each class,
                         optional, default False
      :return:
        if confidence == False:
          list of int or str, resulting prediction on each object
        if confidence == true:
          list of tuple (res, dist)
          :return res: int or str, resulting prediction
          :return dists: dict {str or int: float or int}, proximity to each class
      """
      responses = [classifier.forward_all() for classifier in tqdm(self.classifiers)]
      dists = False
      if method == 'majority':
        votes = [SingleClassifier.only_classes(response) for response in responses]
        res = self.majority_vote(votes)
      elif method == 'weight':
        dists = [SingleClassifier.only_distances(response) for response in responses]
        res = self.weighted_vote(dists)
      else:
        raise KeyError(f'Expected method "majority" or "weight", got "{method}"')
      if confidence:
        if dists:
          return list(zip(res, self.confidence_summing(dists)))
        else:
          dists = [SingleClassifier.only_distances(response) for response in responses]
          return list(zip(res, self.confidence_summing(dists)))
      else:
        return res


    def forward_multiple(self, xs, truncate=True, method='majority', confidence=False):
        """
        predicts classes of give xs using the ensemble

        :param xs: list of collection.Counter, n-gram profile to predict the class of
        :param truncate: bool, whether to apply trucation, optional, default True
        :param method: str ('majority' or 'weight'), nethod of voting to use,
                       optional, default 'majority'
        :param confidence: bool, whether to return proximity score for each class,
                           optional, default False
        :return:
          if confidence == False:
            list of int or str, resulting prediction on each object
          if confidence == true:
            list of tuple (res, dist)
            :return res: int or str, resulting prediction
            :return dists: dict {str or int: float or int}, proximity to each class
        """
        responses = [classifier.forward_multiple(xs, truncate=truncate) for classifier in self.classifiers]
        dists = False
        if method == 'majority':
            votes = [SingleClassifier.only_classes(response) for response in responses]
            res = self.majority_vote(votes)
        elif method == 'weight':
            dists = [SingleClassifier.only_distances(response) for response in responses]
            res = self.weighted_vote(dists)
        else:
            raise KeyError(f'Expected method "majority" or "weight", got "{method}"')
        if confidence:
            if dists:
                return list(zip(res, self.confidence_summing(dists)))
            else:
                dists = [SingleClassifier.only_distances(response) for response in responses]
                return list(zip(res, self.confidence_summing(dists)))
        else:
            return res