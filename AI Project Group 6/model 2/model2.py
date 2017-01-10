from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import learning_curve
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, chi2
import matplotlib.pyplot as plt
import csv

i = 1
corpus = []
Y = []

filename_pos = "data/positive"
filename_neg = "data/negative"

pos_test_questions = "data/test_positive"
neg_test_questions = "data/test_negative.txt"

pos_in_txt = open(filename_pos, "r")
neg_in_txt = open(filename_neg, "r")
pos_test_in_txt = open(pos_test_questions, "r")
neg_test_in_txt = open(neg_test_questions, "r")


test_corpus = []
test_Y = []
for test_content in pos_test_in_txt :
    test_corpus.append(test_content)
    test_Y.append("positive")

for test_content in neg_test_in_txt :
    test_corpus.append(test_content)
    test_Y.append("negative")

for content in neg_in_txt :
    corpus.append(content)
    Y.append("negative")

for content in pos_in_txt :
    corpus.append(content)
    Y.append("positive")

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3))
X = vectorizer.fit_transform(corpus)
test_X = vectorizer.transform(test_corpus)
selector = SelectPercentile(chi2, 35)
X = selector.fit_transform(X, Y)
test_X = selector.transform(test_X)
clf = LogisticRegression()
clf.fit(X,Y)
y_pred = clf.predict(test_X)

precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(test_Y, y_pred, average='macro')
print(f1_score)


X_shuf, Y_shuf = shuffle(test_X, test_Y)
train_sizes, train_scores, test_scores = learning_curve(clf, X_shuf, Y_shuf, train_sizes=[0.6, 0.8, 1])
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.ylim([0.4, 1.2])
plt.show()