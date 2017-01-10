import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.feature_extraction import FeatureHasher

def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

filename_pos = "data/positive"
filename_neg = "data/negative"

# questions_file = "data/q.txt"
# equations_file = "data/eq.txt"
#
# ques_txt = open(questions_file, "r")
# equa_txt = open(equations_file, "r")

pos_test_questions = "data/test_positive"
neg_test_questions = "data/test_negative.txt"

i = 1
corpus = []
Y = []

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


verbs_corpus = []
for each_que in corpus:
    sentences = each_que.split('.')
    verbs = []
    for each_sen in sentences:
        tokens = nltk.word_tokenize(each_sen)
        tagged = nltk.pos_tag(tokens)
        for each_tag in tagged:
            if("VB" in each_tag[1]):
                verbs.append(each_tag[0])
    verbs_corpus.append(str(verbs))

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3))
X = vectorizer.fit_transform(verbs_corpus)
test_X = vectorizer.transform(test_corpus)
selector = SelectPercentile(chi2, 35)
X = selector.fit_transform(X, Y)
test_X = selector.transform(test_X)
clf = LogisticRegression()
clf.fit(X,Y)

verbs_test_corpus = []
for each_test_que in test_corpus:
    #test1 = "Alyssa picked 17 plums and Jason picked 10 plums . Melanie picked 35 pears . How many plums were picked in all ?"
    test1 = each_test_que
    sentences_in_test1 = test1.split('.')
    verbs_test = []
    for each_sentence_in_test1 in sentences_in_test1:
        tokens = nltk.word_tokenize(each_sentence_in_test1)
        tagged = nltk.pos_tag(tokens)
        for each_tag in tagged:
            if ("VB" in each_tag[1]):
                verbs_test.append(each_tag[0])
    verbs_test_corpus.append(str(verbs_test))

X_test = vectorizer.transform(verbs_test_corpus)
X_test = selector.transform(X_test)
y_predict = clf.predict(X_test)
#print(y_predict)
precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(test_Y, y_predict, average='macro')
#print(f1_score)

X_shuf, Y_shuf = shuffle(X_test, test_Y)
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

# plt.legend(loc="best")
# plt.ylim([0.4, 1.2])
# plt.show()
