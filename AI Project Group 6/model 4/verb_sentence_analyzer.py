from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

transferNeg_file = "data/transferNeg"
transferPos_file = "data/transPos"
final_file = "data/final_verbs"
assignTrainVerbs_file = "data/AssignTrainVerbs"

transfer_neg_statements = open(transferNeg_file, "r")
transfer_pos_statements = open(transferPos_file, "r")
final_verb_statements = open(final_file, "r")
assign_train_verbs_statements = open(assignTrainVerbs_file, "r")

test_questions = "data/sampleq.txt"

i = 1
corpus = []
Y = []
test_in_txt = open(test_questions, "r")
sample_test_sen = []
for each_sentence in test_in_txt :
    sample_test_sen.append(each_sentence)

transfer_neg_list = []
y_transfer_neg_list = []
for _content in transfer_neg_statements :
    transfer_neg_list.append(_content)
    y_transfer_neg_list.append("t-")

assign_train_list = []
y_assign_train_list = []
for _content1 in assign_train_verbs_statements :
    assign_train_list.append(_content1)
    y_assign_train_list.append("0")

transfer_pos_list = []
y_transfer_pos_list = []
for _content in transfer_pos_statements :
    transfer_pos_list.append(_content)
    y_transfer_pos_list.append("t+")

final_verb_list = []
y_final_verb_list = []
for _content1 in final_verb_statements :
    final_verb_list.append(_content1)
    y_final_verb_list.append("1")

input_corpus = transfer_neg_list + assign_train_list + transfer_pos_list + y_final_verb_list
# print(len(input_corpus))

Y = y_assign_train_list + y_final_verb_list + y_transfer_pos_list + y_transfer_neg_list
# print(len(Y))

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3))
#X = vectorizer.fit_transform(input_corpus, Y)
#test_X = vectorizer.transform(test_corpus)
# selector = SelectPercentile(chi2, 35)
# X = selector.fit_transform(X, Y)
#test_X = selector.transform(test_X)
clf = LogisticRegression()
clf = Pipeline([('count_vec', vectorizer), ('logistic_regression', clf)])
clf.fit(input_corpus,Y)
joblib.dump(clf, 'trained_model.pkl')
