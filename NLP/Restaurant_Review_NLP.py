# ~~Abhishek K. Singh~~
# Natural Language Processing for a Restaurant Review dataset

# ---Steps---
# 	1. Importing Libraries
# 	2. Reading Dataset
# 	3. Cleaning Text --> Removing unecessary text expression, Covert text to lowercase, Apply Stemming
# 	4. Creation of Bag of Words Model --> Creating Sparse Matrix by Tokenization
# 	5. Using Naive Bayes Classification Algorithm on Bag of words model for prediction


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# print(dataset[90:100])

# Cleaning text
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
	review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(x) for x in review if not x in set(stopwords.words('english'))]
	review = ' '.join(review)
	corpus.append(review)
# print(corpus)

# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Using NaiveBayes on dependent and independent variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix : ")
print(cm)
acc = ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100
print("Accuracy : ", acc, "%")
