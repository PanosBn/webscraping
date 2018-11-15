from IPython import display
import json, requests, webbrowser, re, math, string
import datetime
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
from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


def get_internet_archive():
    # lista me paradeigmata apo kathe istoselida
    timestamps = ['200711', '200811', '200911', '201011', '201111', '201211', '201311', '201411',
                  '201511', '201611', '201711']
    websites = ['guardian.co.uk', 'nytimes.com', 'independent.co.uk', 'https://www.thetimes.co.uk/']
    request_list = {
        'archive.org/wayback/available?url=http://bbc.co.uk/news&timestamp=20091114045636',
        # 'http://archive.org/wayback/available?url=http://economist.com/&timestamp=20091114045636',
        # 'http://archive.org/wayback/available?url=http://guardian.co.uk/news&timestamp=20150925135747',
        # 'http://archive.org/wayback/available?url=http://nytimes.com/&timestamp=20150925135747',
        # 'http://archive.org/wayback/available?url=http://independent.co.uk/&timestamp=20150925135747',
    }
    #Me ta pandas.date_range dimiourgoume ena sunolo apo imerominies gia kathe mera apo to 1998 - 2017
    start_date = datetime.date(1998, 1, 1)
    end_date = datetime.date(2017, 12, 30)

    timestamps = []
    daterange = pd.date_range(start_date, end_date)
    for single_date in daterange:
        dateString = str(single_date.year)+str(single_date.month)+str(single_date.day)+str(120000)
        # print(dateString)
        timestamps.append(dateString)
    http_responses = 0
    snapshots = dict()

    for x in range(1, len(websites)):
        for timestamp in timestamps:
            req = 'http://archive.org/wayback/available?url=' + websites[x] + '/&timestamp=' + timestamp
            # source_website = re.findall("((http:|https:)//[^ \<]*[^ \<\.])", line)
            r = requests.get(req, verify=False)
            print(r)
            if r:
                http_responses = http_responses+1
            print(http_responses)
            data = r.json()
            availabillity = data['archived_snapshots']['closest']['available']

            if (availabillity == True):
                # print("Page exists")
                url = data['archived_snapshots']['closest']['url']
                response = requests.get(url)
                html = response.text
                soup = BeautifulSoup(html, "html.parser")
                for script in soup.findAll('script'):
                    script.extract()
                [s.extract() for s in soup('style')]  # afairesi css kwdika apo selides pou periexoun (Note: Sumvainei mono sti selida twn NYT)

                text = soup.body.getText()  # to text pou exei apomeinei
                stripped_text = text.splitlines()  # spaei me newline

                stripped_text = list(filter(None, stripped_text))
                final_list = []
                for text in stripped_text:
                    text.strip()
                    regx = re.compile(r"\s+")  # merikes ergasies me regex gia na afairethoun leading whitespaces, na sumbitxthoun polla spaces se ena ktl
                    text = regx.sub(repl=" ", string=text)
                    text.strip(" ")
                    regx = re.compile(r"([^\w\s]+)|([_-]+)")
                    text = regx.sub(repl=" ", string=text)
                    sentence_length = (len(text.split()))
                    if (sentence_length > 3):  # Epilegetai na diatirithoun oses eidiseis exoun panw apo 3 lekseis
                        final_list.append(text)
                # for text in final_list:
                #     print(text)
                snapshots[websites[x] + " :" + timestamp] = final_list
            else:
                print("'Tis not here")
    return snapshots


if __name__ == '__main__':

    snapshots = get_internet_archive()
    for line in snapshots:
        print(line)
        print(snapshots[line])

