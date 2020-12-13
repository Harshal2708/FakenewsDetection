import os
import pickle
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     learning_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

%matplotlib inline
warnings.filterwarnings('ignore')

# Creating a list of stopwords
stopwords_list = list(stopwords.words('english'))
stopwords_list
def show_eval_scores(model, test_set, model_name):
    """Function to show to different evaluation score of the model passed
    on the test set.
    
    Parameters:
    -----------
    model: scikit-learn object
        The model whose scores are to be shown.
    test_set: pandas dataframe
        The dataset on which the score of the model is to be shown.
    model_name: string
        The name of the model.
    """
    y_pred = model.predict(test_set['news'])
    y_true = test_set['label']
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('Report for ---> {}'.format(model_name))
    print('Accuracy is: {}'.format(accuracy))
    print('F1 score is: {}'.format(f1))
    print('Precision score is: {}'.format(precision))
    print('Recall score is: {}'.format(recall))
# Importing the datasets
train_data = pd.read_csv('../datasets/train.csv')
valid_data = pd.read_csv('../datasets/valid.csv')
test_data = pd.read_csv('../datasets/test.csv')
train_data.sample(5)
valid_data.sample(5)
print('Train dataset size: {}'.format(train_data.shape))
print('Valid dataset size: {}'.format(valid_data.shape))
print('Test dataset size: {}'.format(test_data.shape))
training_set = pd.concat([train_data, valid_data], ignore_index=True)
print('Training set size: {}'.format(training_set.shape))
training_set.sample(5)
tfidf_V = TfidfVectorizer(stop_words=stopwords_list, use_idf=True, smooth_idf=True)
train_count = tfidf_V.fit_transform(training_set['news'].values)
tfidf_V.vocabulary_
len(tfidf_V.get_feature_names())
lr_pipeline = Pipeline([
    ('lr_TF', TfidfVectorizer(stop_words=stopwords_list, use_idf=True, smooth_idf=True)),
    ('lr_clf', LogisticRegression(random_state=42, n_jobs=-1))
])
lr_pipeline = Pipeline([
    ('lr_TF', TfidfVectorizer(lowercase=False, ngram_range=(1, 5), stop_words=stopwords_list, use_idf=True, smooth_idf=True)),
    ('lr_clf', LogisticRegression(C=1.0, random_state=42, n_jobs=-1))
])
lr_pipeline.fit(training_set['news'], training_set['label'])
show_eval_scores(lr_pipeline, test_data, 'Logistic Regression TFIDF Vectorizer')
nb_pipeline = Pipeline([
    ('nb_TF', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=stopwords_list, use_idf=True, smooth_idf=True)),
    ('nb_clf', MultinomialNB(alpha=2.0))
])
nb_pipeline.fit(training_set['news'], training_set['label'])
show_eval_scores(nb_pipeline, test_data, 'Naive Bayes TFIDF Vectorizer')
lr_voting_pipeline = Pipeline([
    ('lr_TF', TfidfVectorizer(lowercase=False, ngram_range=(1, 5), stop_words=stopwords_list, use_idf=True, smooth_idf=True)),
    ('lr_clf', LogisticRegression(C=1.0, random_state=42, n_jobs=-1))
])
nb_voting_pipeline = Pipeline([
    ('nb_TF', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=stopwords_list, use_idf=True, smooth_idf=True)),
    ('nb_clf', MultinomialNB(alpha=2.0))
])
voting_classifier.fit(training_set['news'], training_set['label'])
show_eval_scores(voting_classifier, test_data, 'Voting Classifier(soft) TFIDF Vectorizer')
pickle.dump(voting_classifier, open(os.path.join('../models', 'voting_classifier_tfidf_vectorizer.pkl'), 'wb'))
