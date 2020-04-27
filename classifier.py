import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sklearn
import nltk

path = '/Users/alinajam/Desktop/winemag-data-130k-v2.csv'

df = pd.read_csv(path, engine = 'python')

parsed_data = df[df.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points', 'price'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

print(parsed_data.head())

dp = parsed_data[['description','points', 'price']]
dp.info()
dp.head()