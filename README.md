# Assignment 2 - Naive Bayes
> Implement Naive Bayes for Sentiment Analysis and Spam Filtering

In this assignment you will implement a Naive Bayes classifier, described in [Chapter 4 of J&M](https://web.stanford.edu/~jurafsky/slp3/4.pdf), to perform sentiment analysis of IMDB reviews and spam detection on sms. You will implement simple **baseline algorithms** to compare your classifiers against. You will also select features to attempt to improve the performance of your classifiers.

You *may* refer to online tutorials in order to implement your classifier and feature extraction, but your implementation must provide the interface described in this document and all sources must be cited in comments.

You will write your code on Google Colab. You should also include a write-up in your colab notebook. The questions you should answer are [at the bottom](#Comparison-and-Writeup).

All your work should be in the Colab notebook. Make sure to leave comments throughout. You may interweave your code and write-up as you see fit/find convenient. If you have concerns or questions, reach out to me on Slack. I am happy to look at your code to provide feedback.

## Evaluation

We have discussed Precision, Recall, Accuracy, and F1 as measures of classifier performance in class. The `datasets` library we used in Assignment 1 to get the IMDB dataset also provides implementations of several evaluation metrics. [You can find the documentation here.](https://huggingface.co/docs/datasets/using_metrics.html) For our purposes, the input for `predictions` and `references` will be a list containing `0`s and `1`s representing the classes:

```python
>>> import datasets
>>> precision = datasets.load_metric("precision")
Downloading: 5.47kB [00:00, 1.81MB/s]
>>> precision.compute(predictions=[1,1,0,1], references=[1,1,1,0])
{'precision': 0.6666666666666666}
```

## Dependencies

This assignment will depend on the `datasets` library to acquire the `imdb` and `sms_spam` datasets. The implementations for metrics will also depend on a library called `sklearn`. You may use any libraries you want to preprocess the data. I recommend using `nltk` to tokenize and process the data.

## Corpora

For sentiment analysis, we will use the [`imdb`](https://huggingface.co/datasets/imdb) dataset. For spam detection, we will use the [`sms_spam`](https://huggingface.co/datasets/sms_spam) dataset.

You can look through the dataset a little on the site. HuggingFace offers great documentation on how to use datasets. [Here is a direct link to how to access the IMDB corpus](https://huggingface.co/docs/datasets/access.html), but I encourage you to look through the documentation yourself.

## Naive Bayes ![Derivative Work](https://img.shields.io/badge/DerivativeWork-%313A55.svg?)

Implement a Naive Bayes classifier as described in [Chapter 4 of J&M](https://web.stanford.edu/~jurafsky/slp3/4.pdf) and lectures. You will need to train your classifier on arbitrary data. For the purposes of this assignment you may store your features as strings. Your probability table could then look something like this:

```python
table = {class1: {
    feature1: count1,
    feature2: count2,
    ...
    featuren: count3
  }, class2: {...}
}
```
> If you feel comfortable doing so, you may also wish to store your features as vectors. In this case a document `d` would b represented as a single vector `v` where the kth element of `v` is 1 if `d` contains the kth feature. This is not required or expected at this point.

You should implement LaPlace (plus one) smoothing to ensure you do not encounter any zero-division errors.

Your final implementation for Naive Bayes should have the following two functions. The code below is a _suggestion_ to help with rapid experimentation of different cleaning techniques and different feature sets.

```python
from typing import Callable, List, Tuple

# Note that a Callable type is a function

feature_list_type = List[str] # if you want to use vectors, you can change this type
# featurizer should take a list of strings (tokens) and return a list of features
featurize_type = Callable[[List[str]], feature_list_type]
# your cleaning function should take a raw string and return a list of strings (tokens)
clean_type = Callable[[str], List[str]]
# your training data should be a list of tuples of (str, int) where the int is the class of the document
train_type = List[Tuple[str, int]]

# this function takes a dataset, a cleaning function, and a list of featurize extraction functions.
# python allows you to treat an uncalled function as a variable.
def learn(data : train_type, clean: clean_type, featurize: List[featurize_type]):
  # learn from data and return a model
  # the following code is just a suggestion. The slides present a more efficient approach.
  for document, cls in data: # iterate over the data
    document = clean(document) # clean the document
    features = []
    for f in featurize:
      features += f(document)
    # update your probability table here using features and classes
  pass
  
def classify(document: str, model, clean: clean_type, featurize: List[featurize_type]):
  # I am not placing restrictions on your model's type
  pass
```

You are welcome to add your own parameters if you document them carefully. For example, if you want to compare Multinomial Naive Bayes to Binary Multinomial Naive Bayes.

## Feature Extraction ![Derivative Work](https://img.shields.io/badge/DerivativeWork-%313A55.svg?)

Implement `bag-of-words` features. This means that an input document (review or sms), has the feature `word_cat` if it contains one or more instances of the word `"cat"` regardless of location or how many times the word occurs. You should additionally implement at least 3 different classes of features. Some possibilities are part-of-speech, punctuation, ngrams, or document length. Think about how you would use these concepts. For example, the important part of document length could be as number of letters, words, sentences, or just some binary measure of whether the document is "long" or "short". You will want to clean your data before you run your feature extraction.

Each of your feature extractors should be an independent function that takes a cleaned and tokenized sentence and returns a list of features. For example, the `bag-of-words` feature extractor could be implemented as follows:

```python
def bag_of_words(cleaned : List[str]):
  # create a set containing all the tokens that are only letters
  # the set will automatically filter out duplicates
  words = {w.lower() for w in cleaned if w.isalpha()]}
  # convert the set to a list before returning.
  return list(words)
```

You are welcome to create more complex feature extracters if you wish. If you do, explain how they work and how to turn them on or off.

> ![Derivative Work](https://img.shields.io/badge/DerivativeWork-%313A55.svg?) means you may freely look up and use code snippets from external sources but you must be able to explain the code. You may not call a library that implements your project and you may not copy-paste an entire project. If you are unsure as to whether you are overstepping the boundaries, ask me on slack!

## Baseline functions

Your baseline classifier should use only the prior probability to predict the class. This means it will always choose the class that is more frequent in the training data.

## Comparison and Writeup

For your two datasets, compare accuracy, precision, and recall to the corresonding baselines. How do your different cleaning methods and features affect performance? Remember, since naive bayes assumes conditional independnce, you don't have to train a new model if you want to remove a feature...

## Grading 
### ? ⭐️ - the number of stars required for an A.

> I will keep track of stars accumulated from assignments instead of some arbitrary percentages. I may offer more stars than are necessary for an A. Anything labelled with a ⚠️ is a hard requirement that I will not make up for with extra credit/bonuses.

- ⚠️ Your code should be well commented and any references you use should be listed in comments. If you forget to keep track of the references, try your best to list them at the bottom.
- ⚠️ Your assignment is submitted within a week of the deadline

- ⭐️

### Bonuses
