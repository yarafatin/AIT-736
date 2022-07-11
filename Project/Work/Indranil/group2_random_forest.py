from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp

import seaborn as sns


import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def lemSentence(sentence):
    token_words = word_tokenize(sentence)
    lem_sentence = []
    for word in token_words:
        lem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        lem_sentence.append(" ")
    return "".join(lem_sentence)

def clean(message, lem=True):
    # Remove ponctuation
    message = message.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    message = message.translate(str.maketrans('', '', string.digits))

    # Remove stop words
    message = [word for word in word_tokenize(message) if not word.lower() in nltk_stopwords]
    message = ' '.join(message)

    # Lemmatization (root of the word)
    if lem:
        message = lemSentence(message)

    return message

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

###To run in local fast START
train = train.head(100000)
#test = test.head(100000)
###To run in local fast END

print (train.shape , test.shape)
#nltk.download('stopwords')
nltk_stopwords = stopwords.words('english')

print('1111 ', datetime.now())
wordnet_lemmatizer = WordNetLemmatizer()

print('2222 ' , datetime.now())

print('3333 ', datetime.now())
train['question_text_cleaned'] = train.question_text.apply(lambda x: clean(x, True))

print('4444 ', datetime.now())


# split  data into training and testing sets of 80:20 ratio

# 80% of test size selected
# random_state is random seed
X_train, X_test, y_train, y_test = train_test_split(train['question_text_cleaned'], train['target'], test_size=0.80, random_state=1)

print('5555 ', datetime.now())

print ('Start Random Forest ')
from sklearn.ensemble import RandomForestClassifier
count_vectorizer = CountVectorizer()
model2 = RandomForestClassifier()

print('6666 ' , datetime.now())
vectorize_model_pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('model', model2)])

print('7777 ' , datetime.now())
vectorize_model_pipeline.fit(X_train, y_train)

print('8888 ' , datetime.now())
predictions2 = vectorize_model_pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions2))

from tabulate import tabulate
data = [[1, 'Accuracy', accuracy_score(y_test, predictions2)],
[2, 'F1 score', f1_score(y_test, predictions2)],
[3, 'Precision', precision_score(y_test, predictions2)],
[4,'Recall', recall_score(y_test, predictions2)]]
print (tabulate(data, headers=["Serial No", "Metrice", "Score"]))

confusion_matrix_insincere = metrics.confusion_matrix(y_test, predictions2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_insincere,
                                            display_labels = [False, True])

cm_display.plot()
plt.show()

test['question_text_cleaned'] = test.question_text.apply(lambda x: clean(x, True))
test['prediction'] = vectorize_model_pipeline.predict(test['question_text_cleaned'])

final = test[['qid','prediction']]
final.set_index('qid', inplace=True)
final.head()

final.to_csv('submission_group2.csv')
