import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from main_backup import BayesModel
import codecs
import pandas as pd
import re
import csv
import random


# df

def gather_test_data():

    perc_samples = 0.01 #to 1% twn seirwn
    samples = []
    colnames = ['sentiment', 'tweet']
    sample = pd.read_csv(
            outputFixedFinal,
            names=colnames,
            encoding='ISO-8859-1',
            header=0,
            skiprows=lambda i: i > 0 and random.random() > perc_samples
    )
    sample = sample.values.tolist()
    for item in sample:
        item[0], item[1] = item[1], item[0]
    return sample

    with open(outputFixedFinal, 'r', encoding="ISO-8859-1") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        for (sentiment, words) in csvreader:
            words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
            tweets.append((words_filtered, sentiment))


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        print(features)
    return features

def train_test_data(tweets):
    tweets_to_tuple = tuple(tweets)
    filtered_tweet = []
    for (words, sentiment) in tweets_to_tuple:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        filtered_tweet.append((words_filtered, sentiment))
    return filtered_tweet

if __name__ == '__main__':

    filename = 'training.1600000.processed.noemoticon.csv'
    output = 'sliced.training.set.csv'
    outputFixed = 'slicedv2.training.set.csv'
    outputFixedV3 = 'slicedv3.training.set.csv'
    outputFixedFinal = 'slicedFinal.training.set.csv'
    delimiter = ','
    with open(filename, 'r', encoding="ISO-8859-1") as csvfile:
        zero = '0'#negative sentiments
        four = '4'#positive sentiments
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        # for line in csvreader:
        #     print(line[5])
        with open(output, 'w', newline='', encoding="ISO-8859-1") as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter)
            writer.writerows(map(itemgetter(0, 5), csvreader))



    # afairw kapoia duplicate entrys poy yparxoyn

    with open(output, 'r', encoding="ISO-8859-1") as csvfile, open(outputFixedV3, 'w', newline='', encoding="ISO-8859-1") as csvfileV2:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        csvwriter = csv.writer(csvfileV2, delimiter=delimiter)
        seen = set()
        for row in csvreader:
            row = tuple(row)
            if row in seen: continue  # skip duplicate
            seen.add(row)
            csvwriter.writerow(row)


    #
    # Sunexizw na katharizw ta tweets afairwntas ta urls kai ta user mentions

    with open(outputFixedV3, 'r', encoding="ISO-8859-1") as csvfile, open(outputFixed, 'w', newline='', encoding="ISO-8859-1") as csvfileV2:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        csvwriter = csv.writer(csvfileV2, delimiter=delimiter)
        for line in csvreader:
            line[1] = re.sub(r"(?:\@|https?\://)\S+", "", line[1])
            csvwriter.writerow(line)
            print(line)
    #
    # printing row count

    with open(outputFixed, 'r', encoding="ISO-8859-1") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        row_count = sum(1 for row in csvreader)
        print(row_count)


    #metrisi plithous positive kai negative tweet

    sum_neg = 0
    sum_pos = 0
    #
    # Allagi tis stilis 0 tou csv arxeioy apo 0 se negative kai 4 se positive gia
    # eukolotero labeling tou sample

    with open(outputFixed, 'r', encoding="ISO-8859-1") as csvfile, open(outputFixedFinal, 'w', newline='',
                                                                        encoding="ISO-8859-1") as csvfileV2:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        csvwriter = csv.writer(csvfileV2, delimiter=delimiter)
        # row_count = sum(1 for row in csvreader)
        # print(row_count)
        zero = '0'
        four = '4'
        for line in csvreader:
            if line[0] == zero:
                line[0] = 'negative'
                csvwriter.writerow(line)
            elif line[0] == four:
                line[0] = 'positive'
                csvwriter.writerow(line)

    with open(outputFixed, 'r', encoding="ISO-8859-1") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        zero = '0'
        four = '4'
        for line in csvreader:
            if line[0] == zero:
                sum_neg = sum_neg + 1
            elif line[0] == four:
                sum_pos = sum_pos + 1

    print("negative:", sum_neg, "Positive:", sum_pos)

    #
    # dialegw random stoixeia apo to csv gia na ta xrisimopoihsw sto testing toy modelou pou tha ftiaksw
    perc_samples = 0.01 #to 1% twn seirwn
    samples = []
    colnames = ['sentiment', 'tweet']
    df = pd.read_csv(
            outputFixedFinal,
            names=colnames,
            encoding='ISO-8859-1',
            header=0,
            skiprows=lambda i: i > 0 and random.random() > perc_samples
    )
    print(df)

if __name__ == '__main__':

    unfiltered_tweets = gather_test_data()
    print(unfiltered_tweets)
    tweets = train_test_data(unfiltered_tweets)
    # print(tweets)
    # for item in sample:
    #     if item[1] == 'negative':
    #         sum_neg = sum_neg + 1
    #     elif item[1] == 'positive':
    #         sum_pos = sum_pos + 1
    # print("pos:",sum_pos,"neg",sum_neg)
    # print(tweets)

    train_test_data()
    word_features = get_word_features(get_words_in_tweets(tweets))
    training_set = nltk.classify.apply_features(extract_features, tweets)
    vector = TfidfVectorizer()
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    #
    tweet = "he is the worst human  i know"
    print(classifier.classify(extract_features(tweet.split())))
