import json, requests, webbrowser, re, math, string
import nltk, pprint
import pickle
from bs4 import BeautifulSoup


class BayesModel(object):

    tweets = []
    test_tweet = []

    def __init__(self):

        pass

    def gather_test_data(self):
        r = requests.get('http://archive.org/wayback/available?url=bbc.co.uk/news&timestamp=20091114045636')
        data = r.json()
        availabillity = data['archived_snapshots']['closest']['available']

        if (availabillity == True):
            print("Page exists")
            url = data['archived_snapshots']['closest']['url']
            response = requests.get(url)
            html = response.text

            soup = BeautifulSoup(html, "html.parser")
            for script in soup.findAll('script'):
                script.extract()

            text = soup.body.getText()
            stripped_text = text.splitlines()

            stripped_text = list(filter(None, stripped_text))
            final_list = []
            for text in stripped_text:
                text.strip()
                sentence_length = (len(text.split()))
                if (sentence_length > 3):
                    final_list.append(text)

            # for text in final_list:
            #     print(text)
            tokenized_text = nltk.word_tokenize(final_list[24])
            # print(tokenized_text)
            # test_tweet = [tokenized_text, 'negative']
            test_tweet = final_list[24]
            # print(test_tweet)
            # print("first and last items of the list are: ", final_list[0], ", ", final_list[-1])

        else:
            print("'Tis not here")

    def train_test_data(self):
        pos_tweets = [('I love this car', 'positive'),
                      ('This view is amazing', 'positive'),
                      ('I feel great this morning', 'positive'),
                      ('I am so excited about the concert', 'positive'),
                      ('He is my best friend', 'positive')]

        neg_tweets = [('I do not like this car', 'negative'),
                      ('This view is horrible', 'negative'),
                      ('I feel tired this morning', 'negative'),
                      ('I am not looking forward to the concert', 'negative'),
                      ('He is my enemy', 'negative')]

        for (words, sentiment) in pos_tweets + neg_tweets:
            words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
            tweets.append((words_filtered, sentiment))

            # print(tweets)
            # print(test_tweet)

    def get_words_in_tweets(self, tweets):
        all_words = []
        for (words, sentiment) in tweets:
            all_words.extend(words)
        return all_words

    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features


if __name__ == '__main__':
    tweets = []  # training set
    test_tweet = []  # negatively labeled BBC headline used for testing

    gather_test_data()
    train_test_data()
    word_features = get_word_features(get_words_in_tweets(tweets))  # feature selection
    training_set = nltk.classify.apply_features(extract_features, tweets)
    # print(word_features)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # f = open('bayes_classifier.pickle', 'wb')
    # pickle.dump(classifier, f)
    # f.close()

    tweet = "John is horrible"
    # print(test_tweet)
    print(classifier.classify(extract_features(tweet.split())))

