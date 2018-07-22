from IPython import display
import json, requests, webbrowser,re,math,string
import nltk, pprint
import praw
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pprint import pprint
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer




nltk.download("stopwords")


# gia xrisi me opoiodipote url
def getSnapshot(domain):
    base = 'http://archive.org/wayback/available?url='
    r = requests.get(base + domain, verify=False)
    data = r.json()

def gather_test_data():
    # lista me paradeigmata apo kathe istoselida
    request_list = {
                    'http://archive.org/wayback/available?url=bbc.co.uk/news&timestamp=20091114045636',
                    # 'http://archive.org/wayback/available?url=economist.com/&timestamp=20091114045636',
                    #  'http://archive.org/wayback/available?url=guardian.co.uk/news&timestamp=20150925135747',
                    # 'http://archive.org/wayback/available?url=nytimes.com/&timestamp=20150925135747',
                    # 'http://archive.org/wayback/available?url=independent.co.uk/&timestamp=20150925135747',
                    }
    for req in request_list:
        r = requests.get(req, verify=False)
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
            [s.extract() for s in soup('style')] #afairesi css kwdika apo selides pou periexoun (Note: Sumvainei mono sti selida twn NYT)

            text = soup.body.getText() #to text pou exei apomeinei
            stripped_text = text.splitlines() #spaei me newline

            stripped_text = list(filter(None, stripped_text))
            final_list = []
            for text in stripped_text:
                text.strip()
                regx = re.compile(r"\s+") #merikes ergasies me regex gia na afairethoun leading whitespaces, na sumbitxthoun polla spaces se ena ktl
                text = regx.sub(repl=" ", string=text)
                text.strip(" ")
                regx = re.compile(r"([^\w\s]+)|([_-]+)")
                text = regx.sub(repl=" ", string=text)
                sentence_length = (len(text.split()))
                if (sentence_length > 3): #Epilegetai na diatirithoun oses eidiseis exoun panw apo 3 lekseis
                    final_list.append(text)
            # for text in final_list:
            #     print(text)
            return final_list
        else:
            print("'Tis not here")

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    filtered_list = []
    for row in text:
        # print(row)
        tokenized = word_tokenize(row)
        cleaned_text = list(filter(lambda x: x not in stop_words, tokenized))
        filtered_list.append(cleaned_text)
    return filtered_list

# def snowball_stemming(text):


def train_test_data():
    tweets = []
    pos_tweets = [('I love this car'),
                  ('This view is amazing'),
                  ('I feel great this morning'),
                  ('I am so excited about the concert'),
                  ('He is my best friend')]

    neg_tweets = [('I do not like this car'),
                  ('This view is horrible'),
                  ('I feel tired this morning'),
                  ('I am not looking forward to the concert'),
                  ('He is my enemy')]

    #Gia to training dokimazw na xrisimopoihsw ta headlines eidisews apo to reddit.com/r/politics
    reddit = praw.Reddit(client_id='sQmRWSARfZy9KQ',
                         client_secret='D0CjeYhWUjuf9qcYNLMGChbwhkA',
                         user_agent='ListenToTheMan ')
    headlines = set()
    for submission in reddit.subreddit('politics').new(limit=None):
        headlines.add(submission.title)
        display.clear_output()

    # print(len(headlines))

    analyser = SentimentIntensityAnalyzer()
    results = []
    for line in headlines:
        pol_score = analyser.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    pprint(results[:5])
    df = pd.DataFrame.from_records(results)
    print(df.head())
    df['label'] = 0
    df.loc[df['compound'] > 0.2, 'label'] = 1
    df.loc[df['compound'] < -0.2, 'label'] = -1
    df.head()

    # df2 = df[['headline', 'label']]
    # df2.to_csv('labeled_politcs_headlines.csv', mode='a', encoding='utf-8', index=False) #antigrafi se csv, ginetai mia mono fora
    print("Positive headlines:\n")
    pprint(list(df[df['label'] == 1].headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(df[df['label'] == -1].headline)[:5], width=200)

    print(df.label.value_counts(normalize=True) * 100)

    fig, ax = plt.subplots(figsize=(8, 8))

    counts = df.label.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Percentage")

    plt.show()

    # Ginetai to idio kai me alli mia 1000ada headlines apo to subreddit /r/news kai tou /r/worldnews giati
    # to prwto batch den itan arketo gia ena axiopisto montelo

    # for submission in reddit.subreddit('news').new(limit=None):
    #     headlines.add(submission.title)
    #     display.clear_output()
    #
    # analyser = SentimentIntensityAnalyzer()
    # results = []
    # for line in headlines:
    #     pol_score = analyser.polarity_scores(line)
    #     pol_score['headline'] = line
    #     results.append(pol_score)
    #
    #     # pprint(results[:5])
    # df = pd.DataFrame.from_records(results)
    # print(df.head())
    # df['label'] = 0
    # df.loc[df['compound'] > 0.2, 'label'] = 1
    # df.loc[df['compound'] < -0.2, 'label'] = -1
    # df.head()
    #
    # df2 = df[['headline', 'label']]
    # df2.to_csv('data/labeled_news_headlines.csv', mode='a', encoding='utf-8', index=False)

    # results = pd.DataFrame([])
    # first_csv = pd.read_csv('data/labeled_politcs_headlines.csv', encoding='utf-8')
    # second_csv = pd.read_csv('data/labeled_news_headlines.csv', encoding='utf-8')
    # results = results.append(first_csv)
    # results = results.append(second_csv)
    # results.to_csv('data/labeled_headlines.csv')

def load_labeled_data(df):
    df = pd.read_csv('data/labeled_headlines.csv', encoding='utf-8')
    print(df.head())
    df = df[df.label != 0] #afairesi twn neutral eidisewn
    # print(df.label.value_counts())

    return df

def vectorize_headlines(df):
    vect = CountVectorizer(max_features=1000, binary=True)
    vector_array = vect.fit_transform(df.headline)
    return vector_array.toarray()

def prepare_training_data(df):
    df_headline = df.headline
    df_label = df.label
    headline_train, headline_test, label_train, label_test = train_test_split(df_headline, df_label, test_size=0.20) #splitting tou 20% tou ogkou twn headlines gia to training
    vect = CountVectorizer(max_features=1000, binary=True)
    headline_train_vector = vect.fit_transform(headline_train)
    headline_test_vector = vect.transform(headline_test)

    balancing = SMOTE()
    headline_train_balanced, label_train_balanced = balancing.fit_sample(headline_train_vector,label_train)
    oversampled_headlines, counts = np.unique(label_train_balanced, return_counts=True)
    print(list(zip(oversampled_headlines, counts)))

    mbayes = MultinomialNB()
    mbayes.fit(headline_train_balanced, label_train_balanced)
    print(mbayes.score(headline_train_balanced, label_train_balanced))

    # actual testing me to testing set pou diaxwrisame
    prediction = mbayes.predict(headline_test_vector)
    print(prediction)
    joblib.dump(mbayes, 'mbayes.pkl')
    joblib.dump(vect, 'vectorizer.pkl')

    # testing me kainourgio external keimeno
    # s1 = ("Senate panel moving ahead with Mueller bill despite McConnell opposition", " 1")
    # headline_test_vector = vect.transform([s1])
    # prediction = mbayes.predict(headline_test_vector.toarray())
    # print(prediction)

    print("Accuracy: {:.4f}%".format(accuracy_score(label_test, prediction) * 100))
    print("F1 Score: {:.4f}".format(f1_score(label_test, prediction) * 100))
    print("FalsePositiveMatrix:\n", confusion_matrix(label_test, prediction))

def load_classifier():
    mbayes = joblib.load('mbayes.pkl')
    vect = joblib.load('vectorizer.pkl')
    s1 = ('Assasination of Failaq Sham fighter by YPG in Afrin, June 24')
    tweet = vect.transform([s1])
    prediction = mbayes.predict(tweet.toarray())
    print(prediction)

# def new_headline_pipeline(headline):
    #do stuff here
    #1 remove leftover hyperlinks and @emails
    #2 tokenize words
    #3 stem tokens
    #4 lemmatize tokens
    #5 remove stopwords
    #6 detokenize and create new string for vectorizing
    # ??
    #profit

if __name__ == '__main__':

    # final_list = []
    # tweets = []
    df = []
    # final_list = gather_test_data()

    #tweets = train_test_data()


    # df = load_labeled_data(df)

    # print(df.label.value_counts())
    # Note. To montelo gia na einai dikaio xreiazetai 50% positive kai negative, either way. Omws epeidi ta
    # headlines pou travame einai tuxaia kai sugkiriaka h analogia auti duskola epitugxanetai.
    # tha prepei na diorthwsoume tin anisoropia me ti methodo tou oversampling
    # http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html


    # vector_array = vectorize_headlines(df)
    # prepare_training_data(df)
    train_test_data()
    # load_classifier()
    # print(vector_array)


    # filtered_words = remove_stopwords(final_list)
    # # print(len(filtered_words))
    # # for x in filtered_words:
    # #     print(x)
    #



        # training_set = remove_stopwords(tweets)
    # print(len(training_set))
    # for x in training_set:
    #     print(x)
    # print(type(training_set))
    # print(filtered_words)

    # for row in final_list:
    #     print(row)