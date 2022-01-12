# Assignment 1 - Tokenization
Implement the Byte-Pair Encoding algorithm and apply it to a large dataset.

In this assignment you will implement the BPE algorithm, described in [Chapter 2 of J&M](https://web.stanford.edu/~jurafsky/slp3/2.pdf). We will then compare it to some other tokenization methods. You will write your code on Google Colab. You should also include a write-up in your colab notebook. The questions you should answer are [at the bottom](#Comparison-and-Writeup).

All your work should be in the Colab notebook. Make sure to leave comments throughout. You may interweave your code and write-up as you see fit/find convenient. If you have concerns or questions, reach out to me on Slack. I am happy to look at your code to provide feedback.

## Dependencies[^1]

For this assignment, we will have to load a corpus. [HuggingFace](https://huggingface.co) hosts a large number of corpora and provides a package called `datasets` that allows us to download and access a corpus easily in our code. After we run our implementation of BPE on the corpus we download, we will also run other existing tokenizers to compare. The other tokenizers are implemented in a package called [`nltk`](https://nltk.org) which contains a number of useful NLP tools.

`nltk` is already pre-installed on Google Colab so we can use it immediately. However, `datasets` is not installed by default. In order to install it, we will use a special syntax available in Colab (and Jupyter) notebooks.

```bash
!pip install datasets
```

The exclamation mark preceding `pip` informs Colab that you wish to run the line in the terminal instead of interpretting it as Python.

## Corpus

We will be using the [IMDB](https://huggingface.co/datasets/imdb) dataset which contains 50,000 movie reviews which were classified by hand as positive or negative. The data is split into a `train` and a `test` set. You will use the `train` section for your **Token Learner** and then run your **Token Segmenter** on the `test` section.

You can look through the dataset a little on the site. HuggingFace offers great documentation on how to use datasets. [Here is a direct link to how to access the IMDB corpus](https://huggingface.co/docs/datasets/access.html), but I encourage you to look through the documentation yourself. It is a (if not the most) vital skill for programming.

We will use the IMDB dataset again next week to see if we can learn to predict whether a review was positive or negative!

## Byte Pair Encoding ![Original Work](https://img.shields.io/badge/OriginalWork-%23ff0077.svg?)

> Remember the ![Original Work](https://img.shields.io/badge/OriginalWork-%23ff0077.svg?) means this section must be done without copying code from external sources. You may still refer to API's and pseudocode. If you are unsure as to whether you are overstepping the boundaries, ask me on slack!

Implement Byte Pair Encoding. Your implementation should have the following functions:

```python
from typing import List

def learn(text: List[str], k : int):
  """Take a list of sentence and apply the Token Learner algorithm.
  Your function should return a list of k tokens in the order they were
  learned from the text
  """"
  pass
  
def segment(text: str, tokens: List[str]):
  """Take a piece of text and return a list of tokens in order.
  eg: segment("lower low", ["l", "o", "w", "e", "r", "_", "lo", "low"])
    returns: ["low", "e", "r", ""_", "low", "_"]
  """
  pass
```

I should be able to copy-paste these functions into my own code, so they should not depend on any global variables. It is ok if they depend on external imports. If you are unsure how to do this, reach out to me.

You may find it helpful to use the `Counter` datastructure from the built-in [`collections`](https://docs.python.org/3/library/collections.html) library.

## Comparison and Writeup

At this stage, we can't objectively test our algorithm because we don't have a *gold standard* for what our tokens should look like. Instead, we can try to compare and contrast the tokenization with other methods. [The `nltk` library provides several tokenizers](https://www.nltk.org/api/nltk.tokenize.html).
* `word_tokenize`
* `wordpunct_tokenize`
* `WhitespaceTokenizer`

For different values of `k`, how does the set of wordtypes and token sequences discovered by BPE compare to the 3 tokenizers above? Is there a high level of overlap? What could you do to your data before learning or segmentation with BPE to make it align better with each of the different approaches?

Write your report in Markdown cells in your notebook. You may interleave code cells to display numbers or statistics.

> Recommendation: Write functions that compare BPE to each tokenizer for different values of `k`.

## Grading 
### 8 ⭐️ - the number of stars required for an A.

> I will keep track of stars accumulated from assignments instead of some arbitrary percentages. I may offer more stars than are necessary for an A. Anything labelled with a ⚠️ is a hard requirement that I will not make up for with extra credit/bonuses.

- ⚠️ Your code should be well commented and any references you use should be listed in comments. If you forget to keep track of the references, try your best to list them at the bottom.
- ⚠️ Your assignment is submitted within a week of the deadline

- ⭐️⭐️ A working Token Learner for BPE
- ⭐️⭐️ A working Token Segmenter for BPE
- ⭐️⭐️⭐️ A detailed comparison of BPE to each of the `nltk` tokenizers. This will involve comparing how many wordtypes and wordtokens are found by each approach, how many types and tokens overlap with BPE, and repeating this process for different values of `k`.
- ⭐️ Your assignment was completed on time.

### Bonuses
- ⭐️ Use Heap's Law/Heardan's Law to choose a good value for `k`. Show your work.
- ⭐️ Implement an improved version of BPE according to your observations and explain why it is an improvement.




[^1]: A dependency is a package containing code written by someone else that you need in order to write or run your own code. Python has a package manager that allows you to install and update packages from a central repository. This makes it easy for people to share code and to make sure everyone has the right version. Several of the dependencies we will use are preinstalled on google colab so we don't have to add them specifically.
