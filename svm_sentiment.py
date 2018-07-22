import sys
import time
import re
import nltk
from sklearn.externals import joblib


def preprocessTweets(tweet):

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    tweet = re.sub('@[^\s]+', '__HANDLE', tweet)

    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = tweet.strip('\'"')

    # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = rpt_regex.sub(r"\1\1", tweet)


    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_join(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(regex_join(replace_parenth(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        tweet = re.sub(regx, ' ' + repl + ' ', tweet)

    tweet = tweet.lower()

    return tweet


# Stemming

def stem(tweet):
    stemmer = nltk.stem.PorterStemmer()
    tweet_stem = ''
    words = [word if (word[0:2] == '__') else word.lower() \
             for word in tweet.split() \
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    tweet_stem = ' '.join(words)
    return tweet_stem


# Predict the sentiment

def predict(tweet, classifier):
    tweet_processed = stem(preprocessTweets(tweet))

    if (('__positive__') in (tweet_processed)):
        sentiment = 1
        return sentiment

    elif (('__negative__') in (tweet_processed)):
        sentiment = 0
        return sentiment
    else:

        X = [tweet_processed]
        sentiment = classifier.predict(X)
        return (sentiment[0])


def main():
    print('Loading the Classifier, please wait....')
    classifier = joblib.load('svmClassifier.pkl')
    print('READY')
    tweet = ' '
    for tweet in sys.stdin:
        print(predict(tweet, classifier))


if __name__ == "__main__":
    main()
