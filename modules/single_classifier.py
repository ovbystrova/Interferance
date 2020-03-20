import numpy as np
from collections import Counter
from modules.distances import radius_distance

def make_counter(commons):
    return Counter({x[0]: x[1] for x in commons})


class SingleClassifier():
    """
    classifies texts as belonging to several classes given true classes

    :param profiles: list of collection.Counter, n-gram profiles
    :param y_true: list of str or int, true classes of each text from profiles
    :param p_length: int, length of truncated profile
    :param classes: list of str or int, possible classes

    Attributes:
      y_true, p_length, profiles (already truncated to p_length), classes
    """


    def __init__(self, profiles, y_true, p_length, classes):
        self.y_true = y_true
        self.p_length = p_length
        self.profiles = self.truncate(profiles, p_length)
        self.classes = classes


    @staticmethod
    def truncate(profiles, p_length):
        """
        truncates a profile to a given length specified by p_length

        :param profiles: list of collection.Counter, n-gram profiles
        :param p_length: int, length of truncated profile,
                         if more than actual profile length, remains unchanged
        :return: list of collection.Counter, text n-gram profiles truncated to
                 the length of p_length
        """
        profiles = [profile.most_common(p_length) for profile in profiles]
        profiles = [make_counter(profile) for profile in profiles]
        return profiles

    def forward_one(self, x):
        """
        predicts the class of a given n-gram profile x

        :param x: collection.Counter, n-gram profile to predict the class of
        :return y: str or int, predicted class
        :return class_dist: dict of {str or int: float}, proximity to each class
        """
        class_dist = {}
        for c in self.classes:
            y_ids = np.where(np.array(self.y_true) == c)  # Выбрать все мемберы этого класса кроме х
            y_c = np.array(self.profiles)[y_ids]

            if x in y_c:
                y_c = np.delete(y_c, np.argwhere(y_c == x))
            distance = radius_distance(x, y_c)
            class_dist[c] = distance

        return min(class_dist, key=class_dist.get), class_dist

    def forward_all(self):
        """
        predicts the class of each profile form profiles

        :return: list of tuples (y, class_dict)
        :type y: str or int, predicted class
        :type class_dist: dict of {str or int: float}, proximity to each class
        """
        return [self.forward_one(x) for x in self.profiles]

    def forward_multiple(self, xs, truncate=True):
        """
        predicts the class of each profile form xs

        :param xs: list of collection.Counter, n-gram profile to predict the class of
        :param truncate: bool, whether to apply trucation, optional, default True
        :return: list of tuples (y, class_dict)
        :type y: str or int, predicted class
        :type class_dist: list of dict of {str or int: float}, proximity to each class
        """
        if truncate:
          xs = self.truncate(xs, self.p_length)
        return [self.forward_one(x) for x in xs]

    @staticmethod
    def only_classes(response):
      """
      retrieves only class predictions from forward_all()

      :param response: list of tuples (y, class_dict)
        :type y: str or int, predicted class
        :type class_dist: dict of {str or int: float}, proximity to each class
      :return: list of str or int, classes
      """
      return [el[0] for el in response]

    @staticmethod
    def only_distances(response, arr=False):
      """
      retrieves only distance predictions from forward_all()

      :param response: list of tuples (y, class_dict)
        :type y: str or int, predicted class
        :type class_dist: dict of {str or int: float}, proximity to each class
      :param arr: bool, if to return list of lists, optional, default False
      :return:
        if arr == False:
          list of dict of {str or int: float}, proximity to each class
        if arr == True:
          :return dists: list of lists, proximity to each class
          :return classes: list of int or str, classes
                           in the same order as in dists
      """
      if not arr:
        return [el[1] for el in response]
      else:
        classes = [el[0] for el in sorted(response[0][1].items())]
        dists = [[el[1] for el in sorted(x[1].items())] for x in response]
        return dists, classes

