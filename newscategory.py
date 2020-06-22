### this is the code for news category 
### challenge for check 24


import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix
import string
from collections import Counter



news = pd.read_csv("news.csv") #Importing data from CSV
print(news['category'].unique())

news = news.fillna(' ')

news['category'] = news.category.map({ 'Wirtschaft': 1, 'Gesundheit': 2, 'Sport': 3, 'Wissenschaft': 4, 'Kultur': 5,'Web': 6 })
# news['text'] = news.text.map(
#     lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
# )

# news.head()


news_known = news[news['category']!=6]
news_unknown = news[news['category']==6]




### training on news_know



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    news_known['text'], 
    news_known['category'], 
    random_state = 1
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])



#Extract features

#Apply bag of words processing to the dataset

from sklearn.feature_extraction.text import CountVectorizer


count_vector = CountVectorizer(min_df=5,max_df=.95)
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)



### Train Multinomial Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

### Generate predictions

predictions = naive_bayes.predict(testing_data)
predictions



###  Evaluate model performance

###  This is a multi-class classification. So, for these evaulation scores, explicitly specify average = weighted

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))



### predict on the unknown news

X_unknown = news_unknown['text']


unknown_data = count_vector.transform(X_unknown)


predictions_unknown = naive_bayes.predict(unknown_data)

predictions_unknown

Counter(predictions_unknown)






###### classify the WirtSchaft into subsection


