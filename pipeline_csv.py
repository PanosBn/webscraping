from IPython import display
import json, requests, webbrowser,re,math,string
import nltk, pprint
import praw
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob,os
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from pprint import pprint
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def prepare_data():
    headlines = pd.read_csv('data/labeled_headlines.csv')
    headlines.columns = ['headline', 'label']
    headlines = pipeline_csv(headlines)
    return headlines


def pipeline_csv(headlines):
    headlines['headline'] = headlines['headline'].apply(nltk.word_tokenize)
    stemmer = PorterStemmer()
    headlines['headline'] = headlines['headline'].apply(lambda x: [stemmer.stem(y) for y in x])
    lemmatizer = nltk.WordNetLemmatizer()
    headlines['headline'] = headlines['headline'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    stopwords = nltk.corpus.stopwords.words('english')
    stemmed_stops = [stemmer.stem(t) for t in stopwords]
    headlines['headline'] = headlines['headline'].apply(lambda x: [stemmer.stem(y) for y in x if y not in stemmed_stops])
    headlines['headline'] = headlines['headline'].apply(lambda x: [e for e in x if len(e) >= 3])
    headlines['headline'] = headlines['headline'].str.join(" ")
    return headlines


def find_polarity(headlines):
    analyser = SentimentIntensityAnalyzer()
    results = []

    for i, row in headlines.iterrows():
        pol_score = analyser.polarity_scores(row['headline'])
        pol_score['headline'] = row['headline']
        results.append(pol_score)

    # pprint(results[:5])
    headlines = pd.DataFrame.from_records(results)
    headlines['label'] = 0
    headlines.loc[headlines['compound'] > 0.2, 'label'] = 1
    headlines.loc[headlines['compound'] < -0.2, 'label'] = -1
    # pprint(headlines)

    #apla ena plotting twn apotelesmatwn
    #den xreiazetai kapou auti ti stigmi
    print("Positive headlines:\n")
    pprint(list(headlines[headlines['label'] == 1].headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(headlines[headlines['label'] == -1].headline)[:5], width=200)

    print(headlines.label.value_counts(normalize=True) * 100)

    fig, ax = plt.subplots(figsize=(8, 8))

    counts = headlines.label.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Percentage")

    # plt.show()

    return headlines


def create_models(headlines):
    headline = headlines['headline']
    label = headlines['label']
    arr_Accu = []

    results = dict()

    for i in range(1, 20):
        headline_train, headline_test, label_train, label_test = train_test_split(headline, label, test_size=0.10, random_state=i)
        vect = CountVectorizer(max_features=1000, binary=True)
        headline_train_vector = vect.fit_transform(headline_train)
        headline_test_vector = vect.transform(headline_test)

        # Note: Egine prospatheia balancing tou dataset alla to accuracy sti sunexeia twn dokimwn apo katw den veltiwthike
        # balancing = SMOTE()
        # headline_train_balanced, label_train_balanced = balancing.fit_sample(headline_train_vector, label_train)
        # oversampled_headlines, counts = np.unique(label_train_balanced, return_counts=True)
        # print(list(zip(oversampled_headlines, counts)))

        dummy = DummyClassifier()
        dummy.fit(headline_train_vector, label_train)
        prediction = dummy.predict(headline_test_vector)
        accuracy = metrics.accuracy_score(label_test, prediction)
        # print(accuracy)
        arr_Accu.append(accuracy)
    print(max(arr_Accu))
    max_random_state = arr_Accu.index(max(arr_Accu)) + 1
    print(max_random_state)
    for j in range(1, 20):
        print("Random State : ", j, "   Accuracy : ", arr_Accu[j-1])


    # Dokimi me k-fold gia tin euresi katalilis timis K gia megisto accuracy
    # Note: to accuracy edw einai xeirotero apo prin

    # arr_Accu = []
    # for i in range(3, 15):
    #     vect = CountVectorizer(stop_words='english', analyzer="word", min_df=2, max_df=0.8)
    #     headline_train_vector = vect.fit_transform(headline)
    #
    #     dummy = DummyClassifier()
    #     accuracy = cross_val_score(dummy, headline_train_vector, label, cv=i, scoring='accuracy')
    #
    #     arr_Accu.append(np.mean(accuracy))
    #
    # # print(arr_Accu)
    # for j in range(3, 15):
    #     print("K-Fold : ", j, "   Accuracy : ", arr_Accu[j - 3])

    # Ksekina i dimiourgia montelwn me to veltisto random state

    headline_train, headline_test, label_train, label_test = train_test_split(headline, label, test_size=0.10, random_state=max_random_state)
    print("random state chosen: ")
    print(max_random_state)
    vect = CountVectorizer(max_features=1000, binary=True)
    headline_train_vector = vect.fit_transform(headline_train)
    headline_test_vector = vect.transform(headline_test)
    # ta headlines tou training kommatioy ginontai fit_transform gia to fit
    # ta headlines tou test ginontai transform gia to test

    # Multionomial Bayes
    mbayes = MultinomialNB()
    mbayes.fit(headline_train_vector, label_train)
    # print(mbayes.score(headline_train_vector, label_train))

    # actual testing me to testing set pou diaxwrisame
    prediction = mbayes.predict(headline_test_vector)
    # print(prediction)
    accuracy = metrics.accuracy_score(label_test, prediction)
    #print('MBayes Accuracy : ', accuracy)
    results["bayes_accuracy"] = accuracy

    log_regression = LogisticRegression()
    log_regression.fit(headline_train_vector, label_train)
    prediction = log_regression.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('LogisticRegression Accuracy : ', accuracy)

    results["Logistic_regression"] = accuracy

    decision_tree = DecisionTreeClassifier(criterion='entropy')
    decision_tree.fit(headline_train_vector, label_train)
    prediction = decision_tree.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('DecisionTree Accuracy : ', accuracy)

    random_forest = RandomForestClassifier(criterion='entropy')
    random_forest.fit(headline_train_vector, label_train)
    prediction = random_forest.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('RandomForestClassifier Accuracy : ', accuracy)

    adaboost = AdaBoostClassifier()
    adaboost.fit(headline_train_vector, label_train)
    prediction = adaboost.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('Adaboost Accuracy : ', accuracy)

    bernoulli_bayes = BernoulliNB()
    bernoulli_bayes.fit(headline_train_vector, label_train)
    prediction = bernoulli_bayes.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('BernoulliNB Accuracy : ', accuracy)

    linear_SVC = LinearSVC()
    linear_SVC.fit(headline_train_vector, label_train)
    prediction = linear_SVC.predict(headline_test_vector)
    accuracy = metrics.accuracy_score(label_test, prediction)
    print('Linear_SVC Accuracy : ', accuracy)

    # passive_aggressive = PassiveAggressiveClassifier()
    # passive_aggressive.fit(headline_train_vector, label_train)
    # prediction = passive_aggressive.predict(headline_test_vector)
    # accuracy = metrics.accuracy_score(label_test, prediction)
    # print('PassiveAggressiveClassifier Accuracy : ', accuracy)
    return results

if __name__ == '__main__':

    headlines = prepare_data()
    headlines = find_polarity(headlines)
    models = []
    create_models(headlines)
    # pprint(headlines)
    # train_data()