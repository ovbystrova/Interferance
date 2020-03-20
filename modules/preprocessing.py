import re
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# import json
from tqdm import tqdm
# from scipy import stats
from collections import Counter
from collections import defaultdict
# from sklearn.model_selection import train_test_split


def collect_ngrams(token_list, n, char=True):
    """
  creates an n-gram list from tokens

  :param token_list: iterable of str, tokens
  :param n: int, length of n-gram
  :param char: bool, whether to create word n-grams or character n-grams,
               optional, default True
  :return: list of str, n-grams
  """
    ngram_list = []
    for i in range(len(token_list)):
        ngram = token_list[i]
        if i + n > len(token_list):
            break
        else:
            for it in range(1, n):
                if char == True:
                    ngram = ngram + token_list[i + it]
                else:
                    ngram = ngram + ' ' + token_list[i + it]
        ngram_list.append(ngram)
    return ngram_list


def make_counter(commons):
    """
  creates a counter from a list of tuples key:count

  :param commons: list of tuple (int or str: int or float), (key: count)
  :return: collections.Counter of the same items
  """
    return Counter({x[0]: x[1] for x in commons})


def freq_ngrams(token_list, n, n_freq, char=True):
    """
  Counter of n-grams from tokens above a certain frequency

  :param token_list: iterable of str, tokens
  :param n: int, length of n-gram
  :param n_freq: int, threshold frequency
  :param char: bool, whether to create word n-grams or character n-grams,
               optional, default True
  :return: collections.Counter of n-grams above a certain frequency
  """
    ngram_list = collect_ngrams(token_list, n, char=char)
    return make_counter(Counter(ngram_list).most_common(n_freq))


def get_dataset(path):
    """
    :param path:
    :return:
    """

    df = pd.read_csv(path, index_col='id')
    df = df.drop(['title', 'date1', 'date2', 'level', 'annotated', 'checked'], axis=1)
    df = df.dropna()

    df['text'] = df.text.apply(lambda x: re.sub('\&[lg]t;', '', x))
    df['len'] = df['text'].apply(lambda x: len(x.split()))
    df['word_tokens'] = df['text'].apply(lambda x: re.findall('\w+', x.lower()))
    df['words_truncated'] = df.word_tokens.apply(lambda x: [x[i] for i in range(len(x)) if i <= 64])

    df['word_unigrams'] = df.words_truncated.apply(lambda x: make_counter(Counter(x).most_common(3000)))
    df['word_bigrams'] = df.words_truncated.apply(lambda x: freq_ngrams(x, 2, 3000, char=False))
    df['word_trigrams'] = df.words_truncated.apply(lambda x: freq_ngrams(x, 3, 3000, char=False))

    df['len_sym'] = df.text.apply(lambda x: len(x))
    df['char_truncated'] = df.text.apply(lambda x: x[:630])

    for i in tqdm(range(3, 11)):
        col_name = f'character {i}-grams'
        df[col_name] = df.char_truncated.apply(lambda x: freq_ngrams(x, i, 3000, char=True))

    df = df.drop(df[df['len'] < 64].index, axis=0)
    df = df.drop(df[df.len_sym < 630].index, axis=0)

    return df


def truncate_dset(df, num_texts):
  """
  Limits texts in each class to a certain number.

  :param df: pandas.DataFrame, dataframe to be limited
  :param num_texts: int, texts number limit

  :return: pandas.DataFrame, limited daraframe
  """
  langs = defaultdict(int)
  curr_native = df.native[1]
  langs[curr_native] += 1
  limited = df[df.index == 1]
  for i in list(df.index)[1:]:
    curr_text = df[df.index == i]
    if langs[curr_text.native[i]] >= num_texts:
      continue
    else:
      curr_native = curr_text.native[i]
      langs[curr_text.native[i]] += 1
      limited = limited.append(curr_text)

  return limited


def balanced_datasets(df):
    """
    :param df:
    :return:
    """

    classes = Counter(df.native)
    df['num_texts'] = df.native.apply(lambda x: classes[x])

    df_90texts = truncate_dset(df[df.num_texts >= 90], 90)
    df_400texts = truncate_dset(df[df.num_texts >= 400], 400)

    return df_90texts, df_400texts




if __name__ == '__main__':
    PATH = 'data/original_texts.csv'
