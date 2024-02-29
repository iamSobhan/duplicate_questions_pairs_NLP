#importing libraries
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz


#importing the model
cv = pickle.load(open("cv.pkl", "rb"))


def test_common_words(q1, q2):
    #split sentence q1 into words, convert to lowercase, and create a set of unique words
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    #split sentence q2 into words, convert to lowercase, and create a set of unique words
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    #return the count of common words between the two sets of words
    return len(w1 & w2)


def test_total_words(q1, q2):
    #split sentence q1 into words, convert to lowercase, and create a set of unique words
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    #split sentence q2 into words, convert to lowercase, and create a set of unique words
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    #return the total count of words in both sentences combined
    return (len(w1) + len(w2))


def test_fetch_token_features(q1, q2):
    # define a small constant to avoid division by zero
    SAFE_DIV = 0.0001

    # import stopwords from the NLTK library for English language
    STOP_WORDS = stopwords.words("english")

    # initialize a list of token features with zeros
    token_features = [0.0] * 8

    # converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    # if either question has no tokens, return the list of token features with zeros
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # get the count of common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # get the count of common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # get the count of common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    # calculate various token features
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # check if the last word of both questions is the same
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # check if the first word of both questions is the same
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    # return the list of calculated token features
    return token_features


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3  # initialize a list to store length features, initially filled with zeros

    # converting the Sentence into Tokens:
    q1_tokens = q1.split()  # split the first sentence into tokens
    q2_tokens = q2.split()  # split the second sentence into tokens

    if len(q1_tokens) == 0 or len(
            q2_tokens) == 0:  # if either sentence has no tokens, return the list of length features with zeros
        return length_features

    # absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))  # calculate the absolute difference in token lengths

    # average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2  # calculate the average token length of both sentences

    strs = list(distance.lcsubstrings(q1, q2))  # calculate the longest common substring between the two sentences
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)  # calculate the ratio of the length of the longest common substring to the minimum length of the two sentences

    return length_features  # return the list of calculated length features


def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4  # initialize a list to store fuzzy features, initially filled with zeros

    # calculate different fuzzy matching ratios between the two input sentences

    # fuzz_ratio: Compares entire strings, returns a ratio between 0 and 100 indicating similarity
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio: Compares partial strings, returns a ratio between 0 and 100 indicating similarity
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio: Compares sorted token lists, returns a ratio between 0 and 100 indicating similarity
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio: Compares token sets, returns a ratio between 0 and 100 indicating similarity
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features  # return the list of calculated fuzzy features


# preprocessing

def preprocessing(p):
    # converting the input to lowercase and removing leading/trailing whitespaces
    p = str(p).lower().strip()

    # replacing special charecters with their string equivalents

    p = p.replace("%", " percent ")
    p = p.replace("$", " dollar ")
    p = p.replace("₹", " rupee ")
    p = p.replace("€", " euro")
    p = p.replace("@", " at ")

    # the pattern math appears around 900 times in the whole dataset
    p = p.replace("[math]", "")

    # replacing some numbers with string equivalents
    p = p.replace(',000,000,000 ', 'b ')
    p = p.replace(',000,000 ', 'm ')
    p = p.replace(',000 ', 'k ')
    p = re.sub(r'([0-9]+)000000000', r'\1b', p)
    p = re.sub(r'([0-9]+)000000', r'\1m', p)
    p = re.sub(r'([0-9]+)000', r'\1k', p)

    # decontracting words
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    p_decontracted = []

    for word in p.split():
        if word in contractions:
            word = contractions[word]

        p_decontracted.append(word)

    p = ' '.join(p_decontracted)
    p = p.replace("'ve", " have")
    p = p.replace("n't", " not")
    p = p.replace("'re", " are")
    p = p.replace("'ll", " will")

    # removing HTML tags
    p = BeautifulSoup(p)
    p = p.get_text()

    # removing punctuations
    pattern = re.compile("\W")
    p = re.sub(pattern, " ", p).strip()

    return p


def query_point_creator(q1, q2):
    input_query = []  # initialize an empty list to store features

    # preprocess the input sentences
    q1 = preprocessing(q1)
    q2 = preprocessing(q2)

    # fetch basic features

    # append lengths of q1 and q2
    input_query.append(len(q1))
    input_query.append(len(q2))

    # append number of words in q1 and q2
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    # fetch and append common word features
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features and append
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features and append
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features and append
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # create bag-of-words (BOW) features for q1 and q2
    q1_bow = cv.transform([q1]).toarray()  # convert q1 into bag-of-words representation
    q2_bow = cv.transform([q2]).toarray()  # convert q2 into bag-of-words representation

    # concatenate all features along columns to create the final input query point
    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))  # stack the arrays horizontally

