from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile

transferNeg_file = "data/transferNeg"
transferPos_file = "data/transPos"
final_file = "data/final_verbs"
assignTrainVerbs_file = "data/AssignTrainVerbs"
output_verbs = "data/output_verbs"

# questions_file = "q.txt"
# equations_file = "eq.txt"

transfer_neg_statements = open(transferNeg_file, "r")
transfer_pos_statements = open(transferPos_file, "r")
final_verb_statements = open(final_file, "r")
assign_train_verbs_statements = open(assignTrainVerbs_file, "r")
output_file = open(output_verbs, "w")
test_questions = "data/sampleq.txt"

i = 1
corpus = []
Y = []
# in_txt = open(filename_neg, "r")
test_in_txt = open(test_questions, "r")

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

verbs_neg_corpus = []
for each_sen in transfer_neg_list:
    #sentences = each_que.split('.')
    verbs = []
    #for each_sen in sentences:
    tokens = nltk.word_tokenize(each_sen)
    tagged = nltk.pos_tag(tokens)
    for each_tag in tagged:
        if("VB" in each_tag[1]):
            #print("found VB tag - " + str(each_tag))
            verbs.append(each_tag[0])
    #print(verbs)
    verbs_neg_corpus.append(str(verbs))

# print(len(verbs_neg_corpus))
# print(len(corpus))
# print(verbs_neg_corpus[0])

verbs_pos_corpus = []
for each_sen in transfer_pos_list:
    #sentences = each_que.split('.')
    verbs = []
    #for each_sen in sentences:
    tokens = nltk.word_tokenize(each_sen)
    tagged = nltk.pos_tag(tokens)
    for each_tag in tagged:
        if("VB" in each_tag[1]):
            #print("found VB tag - " + str(each_tag))
            verbs.append(each_tag[0])
    #print(verbs)
    verbs_pos_corpus.append(str(verbs))

# print(len(verbs_pos_corpus))
# print(len(corpus))
# print(verbs_pos_corpus[0])

verbs_assign_corpus = []
for each_sen in assign_train_list:
    #sentences = each_que.split('.')
    verbs = []
    #for each_sen in sentences:
    tokens = nltk.word_tokenize(each_sen)
    tagged = nltk.pos_tag(tokens)
    for each_tag in tagged:
        if("VB" in each_tag[1]):
            #print("found VB tag - " + str(each_tag))
            verbs.append(each_tag[0])
    #print(verbs)
    verbs_assign_corpus.append(str(verbs))

# print(len(verbs_assign_corpus))
# print(len(corpus))
# print(verbs_assign_corpus[0])

verbs_final_corpus = []
for each_sen in final_verb_list:
    #sentences = each_que.split('.')
    verbs = []
    #for each_sen in sentences:
    tokens = nltk.word_tokenize(each_sen)
    tagged = nltk.pos_tag(tokens)
    for each_tag in tagged:
        if("VB" in each_tag[1]):
            #print("found VB tag - " + str(each_tag))
            verbs.append(each_tag[0])
    #print(verbs)
    verbs_final_corpus.append(str(verbs))

# print(len(verbs_final_corpus))
# print(len(corpus))
# print(verbs_final_corpus[0])

verbs_corpus = verbs_assign_corpus + verbs_final_corpus + verbs_pos_corpus + verbs_neg_corpus
print(len(verbs_corpus))

Y = y_assign_train_list + y_final_verb_list + y_transfer_pos_list + y_transfer_neg_list
print(len(verbs_corpus))

for i in range(len(verbs_corpus)):
    output_file.write(str(verbs_corpus[i])+"\t"+Y[i]+"\n")

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3))
X = vectorizer.fit_transform(verbs_corpus)
#test_X = vectorizer.transform(test_corpus)
selector = SelectPercentile(chi2, 35)
X = selector.fit_transform(X, Y)
#test_X = selector.transform(test_X)
clf = LogisticRegression()
clf.fit(X,Y)

verbs_test_corpus = []
for each_test_que in test_in_txt:
     #test1 = "Alyssa picked 17 plums and Jason picked 10 plums . Melanie picked 35 pears . How many plums were picked in all ?"
    test1 = each_test_que
    sentences_in_test1 = test1.split('.')
    verbs_test = []
    for each_sentence_in_test1 in sentences_in_test1:
        tokens = nltk.word_tokenize(each_sentence_in_test1)
        tagged = nltk.pos_tag(tokens)
        for each_tag in tagged:
            if ("VB" in each_tag[1]):
#                 # print("found VB tag - " + str(each_tag))
                 verbs_test.append(each_tag[0])
#             # print(verbs)
    verbs_test_corpus.append(str(verbs_test))

X_test = vectorizer.transform(verbs_test_corpus)
X_test = selector.transform(X_test)
y_predict = clf.predict(X_test)
print(y_predict)