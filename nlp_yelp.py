import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import string

yelp = pd.read_csv('yelp.csv')

print(yelp.head())
print(yelp.info())
print(yelp.describe())

yelp['text length'] = yelp['text'].apply(len)
print(yelp.head())

g = sns.FacetGrid(yelp, col = 'stars')
g.map(plt.hist, 'text length')
plt.show()

sns.boxplot(x = 'stars', y ='text length', data = yelp, palette = 'rainbow')
plt.show()

sns.countplot(x = 'stars', data = yelp, palette = 'rainbow')
plt.show()

stars = yelp.groupby('stars').mean()

print(stars.corr())

sns.heatmap(stars.corr(), cmap = 'coolwarm', annot = True)
plt.tight_layout()
plt.show()

yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]

X = yelp_class['text']
y = yelp_class['stars']

ctv = CountVectorizer()

X = ctv.fit_transform(X)

T_train, T_test, l_train, l_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

nb = MultinomialNB()

nb.fit(T_train, l_train)

predictions = nb.predict(T_test)

print(confusion_matrix(l_test, predictions))
print(classification_report(l_test, predictions))

pipeline = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mnb', MultinomialNB())
])

X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train, y_train)

p_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, p_pred))
print('\n')
print(classification_report(y_test, p_pred))