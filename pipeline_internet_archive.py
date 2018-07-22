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
    websites = ['guardian.co.uk', 'nytimes.com', 'independent.co.uk']
    request_list = {
                    'archive.org/wayback/available?url=http://bbc.co.uk/news&timestamp=20091114045636',
                    # 'http://archive.org/wayback/available?url=http://economist.com/&timestamp=20091114045636',
                    # 'http://archive.org/wayback/available?url=http://guardian.co.uk/news&timestamp=20150925135747',
                    # 'http://archive.org/wayback/available?url=http://nytimes.com/&timestamp=20150925135747',
                    # 'http://archive.org/wayback/available?url=http://independent.co.uk/&timestamp=20150925135747',
                    }
    snapshots = dict()
    for x in range(1, len(websites)):
        # for req in request_list:
            for i in range(1, len(timestamps)):
                req = 'http://archive.org/wayback/available?url='+websites[x]+'/&timestamp='+timestamps[i]+'14045636'
                # source_website = re.findall("((http:|https:)//[^ \<]*[^ \<\.])", line)
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
                    snapshots[websites[x] + " :" + timestamps[i]] = final_list
                else:
                    print("'Tis not here")
    return snapshots

if __name__ == '__main__':

   snapshots = get_internet_archive()
   for line in snapshots:
       print(line)
       print(snapshots[line])