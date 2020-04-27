import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sklearn
import nltk
from sklearn.naive_bayes import GaussianNB, MultinomialNB

path = '/Users/alinajam/Desktop/winemag-data-130k-v2.csv'

df = pd.read_csv(path, engine = 'python')

parsed_data = df[df.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

print(parsed_data.head())

dp = parsed_data[['description','points']]
dp.info()
dp.head()

def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2
    elif points >= 88 and points < 92:
        return 3
    elif points >= 92 and points < 96:
        return 4
    else:
        return 5


dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
dp.head()

X = dp['description']
y = dp['points_simplified']

vectorizer = TfidfVectorizer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
vectorizer.fit(X)
X = vectorizer.transform(X)


gNB = GaussianNB()
mnNB = MultinomialNB()

gNB.fit(X_train. astype('int'), y_train.astype('int'))
predictions = gNB.predict(X_test)

print(classification_report(y_test, predictions))