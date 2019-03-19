 # Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
# Importing the dataset
dataset = pd.read_csv('/Profiles/lsethia/Downloads/spam-filter/org.csv')
dataset = dataset[0:5729]
dataset=dataset.loc[:,'text':'spam']
dataset["text"].fillna("Not text", inplace = True) 
dataset["spam"].fillna("0", inplace = True) 
print(dataset)
# Cleaning the texts
corpus = []
for i in range(0, 5729):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(len(corpus))    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.loc[:,'spam'].values
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
c=0
d=len(y_test)
y_pred = classifier.predict(x_test)
for i in range(0,d):
    if (y_pred[i]==y_test[i]):
        c=c+1
print(c)
print(len(y_test))
print(c/len(y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)